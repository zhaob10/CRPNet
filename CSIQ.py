import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from scipy import stats
import loader_patch一整张 as loader
import my_config_whole as config
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
configer = tf.ConfigProto()

# 是否使用learning_rate
learning_r_decay = config.learning_r_decay
learning_rate_base = config.learning_rate_base
decay_rate = config.decay_rate

# 顺序
ORDER = config.ORDER

TRAIN_DATA_NUM = config.TRAIN_DATA_NUM
TEST_DATA_NUM = config.TEST_DATA_NUM
# 训练集参数
batch_size_train = config.batch_size_train

capacity_train = config.capacity_train
min_after_dequeue_train = config.min_after_dequeue_train
# 测试集参数
batch_size_test = config.batch_size_test
capacity_test = config.capacity_test
min_after_dequeue_test = config.min_after_dequeue_test

# data地址
train_data_dir = config.train_data_dir
test_data_dir = config.test_data_dir
# 图像大小
img_size = config.img_size

tf.reset_default_graph()

print('data order: %s' % ORDER)

patch_per_img = 128  # 128channel 出128个分

# 注意这里的改动
input = tf.placeholder(tf.float32, [batch_size_train * 256, 32, 32, 1])
label = tf.placeholder(tf.float32, [batch_size_train * patch_per_img, 1])  # 为了后分割加数据量罢了
is_train = tf.placeholder(tf.bool, name='is_train')
input_step2 = tf.placeholder(tf.float32, [batch_size_train , 128])#第一阶段预测的分数

global_step = tf.Variable(0, trainable=False)

init_learning_rate = 0.01  # 这里记得改

learning_rate = tf.train.exponential_decay(init_learning_rate,global_step=global_step,decay_steps=982, decay_rate=0.98, staircase=True)
add_global = global_step.assign_add(1)


class net(object):
    def __init__(self):
        self.is_training = is_train

        def split_channel(patch):  # axis=2给他unstack，然后把这些重新stack 就完成了channel维度的拆分重组
            channel_list = tf.unstack(patch, axis=2)
            return tf.unstack(channel_list, axis=0)

        def deal_per_patch(patch):  # [256 ,4*4, channel] 如何让他变成以channel为导向的呢？
            tire1, tire2, tire3, tire4 = tf.split(patch, [32, 32, 64, 128], axis=0)  # [64,4*4,64] 分成四份
            # 64*16*64 所以看起来只取top64确实好些。 把别的掺杂进来以后分数糟糕了不少 。这么说其实还是该筛选。或者给低patch足够低的权重
            tire1 = split_channel(tire1)  # 32*16 1:1 = 512 下次最高的32要最高的比例 一锤定音1：1
            tire2 = split_channel(tire2)  # 64*16 4:1 = 256
            tire3 = split_channel(tire3)  # 96*16 6:1 = 256
            return tf.reshape(tire1, [128, 32 * 4 * 4]), tf.reshape(tire2, [128, 32 * 4 * 4]), tf.reshape(tire3, [128, 64 * 4 * 4])

        def conv_net():
            with arg_scope([layers.conv2d], padding='SAME',
                           normalizer_fn=layers.batch_norm, normalizer_params={"is_training": self.is_training}
                           ):
                net = layers.conv2d(input, 32, [3, 3], [1, 1], scope='convd1')
                net = layers.conv2d(net, 32, [3, 3], [1, 1], scope='convd2')
                net = layers.conv2d(net, 32, [3, 3], [1, 1], scope='convd3')
                net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

                net = layers.conv2d(net, 64, [3, 3], [1, 1], scope='convd4')
                net = layers.conv2d(net, 64, [3, 3], [1, 1], scope='convd5')
                net = layers.conv2d(net, 64, [3, 3], [1, 1], scope='convd6')
                net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

                net = layers.conv2d(net, 128, [3, 3], [1, 1], scope='convd7')
                net = layers.conv2d(net, 128, [3, 3], [1, 1], scope='convd8')
                net = layers.conv2d(net, 128, [3, 3], [1, 1], scope='convd9')
                net = tf.nn.max_pool(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
                net_list = tf.split(net, batch_size_train, axis=0)  # 这里是4个 [256*16*64]

                # 或者权重能否加在内部？
                for i in range(batch_size_train):
                    tmp = deal_per_patch(net_list[i])  # 在这里就把featureMap展平了
                    if i == 0:
                        tire1 = tmp[0]
                        tire2 = tmp[1]
                        tire3 = tmp[2]
                    else:
                        tire1 = tf.concat([tire1, tmp[0]], axis=0)
                        tire2 = tf.concat([tire2, tmp[1]], axis=0)
                        tire3 = tf.concat([tire3, tmp[2]], axis=0)

                return tire1, tire2, tire3

        self.output = conv_net()  # [8*128,32*16]*3

        self.tire1 = layers.fully_connected(self.output[0], 512, normalizer_fn=layers.batch_norm,normalizer_params={"is_training": self.is_training})
        self.tire2 = layers.fully_connected(self.output[1], 256, normalizer_fn=layers.batch_norm,normalizer_params={"is_training": self.is_training})
        self.tire3 = layers.fully_connected(self.output[2], 256, normalizer_fn=layers.batch_norm,normalizer_params={"is_training": self.is_training})
        self.output = tf.concat([self.tire1, self.tire2, self.tire3], axis=1)  # 拼接起来
        self.output = layers.fully_connected(self.output, 1024, normalizer_fn=layers.batch_norm,normalizer_params={"is_training": self.is_training})
        self.output = layers.fully_connected(self.output, 1, activation_fn=None)

        self.loss = tf.losses.mean_squared_error(label, self.output)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,
                                                                                         global_step=global_step)


mynet = net()



class net_weight(object):
    def __init__(self):
        self.is_training = is_train

        def precess_weight(weight):
            weight_sig = tf.multiply( tf.nn.sigmoid(weight) , 2)
            result = weight_sig
            for i in range(batch_size_train - 1):
                result = tf.concat([result, weight_sig], axis=0)
            return result

        def weight_net():
            with arg_scope([layers.conv2d], padding='SAME',normalizer_fn=layers.batch_norm, normalizer_params={"is_training": self.is_training}):
                weight = tf.fill([1, 128], 1.0)
                # 假定给了false就不会有变化
                weight = layers.fully_connected(weight, 128, normalizer_fn=layers.batch_norm,normalizer_params={"is_training": self.is_training},scope="fc1")
                weight = layers.fully_connected(weight, 128, normalizer_fn=layers.batch_norm,normalizer_params={"is_training": self.is_training}, scope="fc2")
                weight = layers.fully_connected(weight, 128, activation_fn=None,scope="fc3")
                weight = precess_weight(weight)
                res = tf.multiply(input_step2,weight)
                return res,weight


        self.output = weight_net()
        self.outs = self.output
        self.output = tf.reshape(self.output[0],[batch_size_train*patch_per_img , 1])
        self.loss = tf.losses.mean_squared_error(label, self.output)

        update_ops_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="fc1|fc2|fc3")
        with tf.control_dependencies(update_ops_w):
            self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss,global_step=global_step)

weightnet=net_weight()

train = True
test = False

data_train = loader.get_traindata(train_data_dir, ORDER, TRAIN_DATA_NUM, TEST_DATA_NUM, batch_size_train,
                                  capacity_train, min_after_dequeue_train)
data_test = loader.get_testdata(test_data_dir, ORDER, TRAIN_DATA_NUM, TEST_DATA_NUM, batch_size_test, batch_size_test,
                                min_after_dequeue_test)
saver = tf.train.Saver()


def split_img(batch, image_whole, ranked):  # 将每幅图都按照降序排列
    img_split = np.zeros([256 * batch, 32, 32, 1])
    a = 0
    for k in range(batch):
        for i in range(16):  # 行
            for j in range(16):  # 列
                img_split[a, :, :, :] = image_whole[k, i * 32: (i + 1) * 32, j * 32:(j + 1) * 32, :]
                a = a + 1
    img_res = np.zeros([256 * batch, 32, 32, 1])
    q = 0
    for l in range(batch):
        for p in range(256):
            img_res[q, :, :, :] = img_split[256 * l + ranked[l, p], :, :, :]
            q = q + 1
    return img_res


def rand(input, idx):  # 打乱顺序
    shuffle = input
    for i in range(batch_size_train * 64):
        shuffle[i, :] = input[idx[i], :]
    return shuffle


with tf.Session(config=configer) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    #算分
    def calScore1(img_t_total,dmos_t_total,weight_t_total):
        pre_total = []
        dmos_total = []
        for j in range(16):  #
            img_t = img_t_total[j * batch_size_train:(j + 1) * batch_size_train, :, :, :]
            dmos_t = dmos_t_total[j * batch_size_train:(j + 1) * batch_size_train, :]
            weight_t = weight_t_total[j * batch_size_train:(j + 1) * batch_size_train, :]
            dmos_t = dmos_t.repeat(patch_per_img).reshape([-1, 1])

            ranked_t = np.argsort(-weight_t).reshape([batch_size_train, 256])
            img_patches_t = split_img(batch_size_train, img_t, ranked_t)  # 可以切成256patch

            pre = sess.run([mynet.output],
                           feed_dict={input: img_patches_t, label: dmos_t, is_train: False})

            pre = np.asarray(pre).reshape([batch_size_train, 128])
            if j == 0:
                pre_total = pre
                dmos_total = dmos_t
            else:
                pre_total = np.concatenate((pre_total, pre), axis=0)
                dmos_total = np.concatenate((dmos_total, dmos_t), axis=0)

        src = stats.spearmanr(np.mean(pre_total,axis=1), np.mean(dmos_total.reshape([-1,128]) , axis = 1))[0]
        plc = stats.pearsonr(np.mean(pre_total,axis=1), np.mean(dmos_total.reshape([-1,128]) , axis = 1))[0]
        test_los = sess.run([mynet.loss],
                             feed_dict={input: img_patches_t, label: dmos_t, is_train: False})
        print('i = ', i, 'test src = ', src, 'test plc =', plc, 'train_loss=', test_los)
        return src,test_los[0]

    def calScore2(img_t_total,dmos_t_total,weight_t_total):
        pre_total = []
        dmos_total = []
        for j in range(16):  #
            img_t = img_t_total[j * batch_size_train:(j + 1) * batch_size_train, :, :, :]
            dmos_t = dmos_t_total[j * batch_size_train:(j + 1) * batch_size_train, :]
            weight_t = weight_t_total[j * batch_size_train:(j + 1) * batch_size_train, :]
            dmos_t = dmos_t.repeat(patch_per_img).reshape([-1, 1])

            ranked_t = np.argsort(-weight_t).reshape([batch_size_train, 256])
            img_patches_t = split_img(batch_size_train, img_t, ranked_t)  # 可以切成256patch

            pre = sess.run([mynet.output],feed_dict={input: img_patches_t, label: dmos_t, is_train: False})
            pre = np.asarray(pre).reshape([batch_size_train, 128])
            weightedScore = sess.run([weightnet.output], feed_dict={input_step2: pre, label: dmos_t, is_train: False})
            weightedScore = np.asarray(weightedScore).reshape([-1,128])
            if j == 0:
                pre_total = weightedScore
                dmos_total = dmos_t
            else:
                pre_total = np.concatenate((pre_total, weightedScore), axis=0)
                dmos_total = np.concatenate((dmos_total, dmos_t), axis=0)

        src = stats.spearmanr(np.mean(pre_total,axis=1), np.mean(dmos_total.reshape([-1,128]) , axis = 1))[0]
        plc = stats.pearsonr(np.mean(pre_total,axis=1), np.mean(dmos_total.reshape([-1,128]) , axis = 1))[0]
        test_los = sess.run([weightnet.loss],
                             feed_dict={input_step2: pre, label: dmos_t, is_train: False})
        print('i = ', i, 'test src = ', src, 'test plc =', plc, 'train_loss=', test_los)
        return src, test_los[0]

    if True:
        los_tmp = 2800
        src_tmp = 0.969
        for i in range(34001):
            img, dmos, weight = sess.run([data_train[0], data_train[1], data_train[2]])
            dmos = dmos.repeat(patch_per_img).reshape([-1, 1])
            ranked = np.argsort(-weight).reshape([batch_size_train, 256])
            img_patches = split_img(batch_size_train, img, ranked)

            if i<32000:
                opt = sess.run([mynet.train_op], feed_dict={input: img_patches, label: dmos, is_train: True})
                if (i % 200 == 0 ):
                    # 一次性拿出test全集
                    img_t_total, dmos_t_total, weight_t_total = sess.run([data_test[0], data_test[1], data_test[2]])
                    dmos_t_total = dmos_t_total.reshape([-1, 1])
                    #输出是  [batchSize*128,1]
                    statics = calScore1(img_t_total, dmos_t_total, weight_t_total)
                    if (i >= 20000 ):
                        if (statics[1] < los_tmp and statics[0] > 0.97):
                            saver.save(sess, r'F:\zch\plan2020\logs\2step\step1\one_init_los/train.ckpt')
                            los_tmp = statics[1]

                        if (statics[0] > src_tmp):
                            saver.save(sess, r'F:\zch\plan2020\logs\2step\step1\one_init_src/train.ckpt')
                            src_tmp = statics[0]
            else:
                pre = sess.run([mynet.output], feed_dict={input: img_patches, label: dmos, is_train: False})  # 拿到预测得分
                pre = np.asarray(pre).reshape([batch_size_train, 128])
                # 只训练这两层
                opt_w = sess.run([weightnet.train_op], feed_dict={input_step2: pre, label: dmos, is_train: True})
                if (i % 200 == 0 ):
                    # 一次性拿出test全集
                    img_t_total, dmos_t_total, weight_t_total = sess.run([data_test[0], data_test[1], data_test[2]])
                    dmos_t_total = dmos_t_total.reshape([-1, 1])
                    #输出是  [batchSize*128,1]
                    statics = calScore2(img_t_total, dmos_t_total, weight_t_total)
                    if (i >= 20000 ):
                        if (statics[1] < los_tmp and statics[0] > 0.97):
                            saver.save(sess, r'F:\zch\plan2020\logs\2step\step2/one_init_los/train.ckpt')
                            los_tmp = statics[1]

                        if (statics[0] > src_tmp):
                            saver.save(sess, r'F:\zch\plan2020\logs\2step\step2\one_init_src/train.ckpt')
                            src_tmp = statics[0]



    coord.request_stop()
    coord.join(threads)

