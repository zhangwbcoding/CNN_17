import tensorflow as tf
import numpy as np
import utils
import os
import time
import data


class CNN_17(object):
    def __init__(self, batch_size=800, image_size=32, channels=18, datasize=100000, dtype=np.float32):

        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self.datasize = datasize

        # 输入
        self.x = tf.placeholder(dtype, [None, image_size, image_size, channels], name='x')
        self.label = tf.placeholder(np.int32, [None, 17], name='label')
        self.keep_prob = tf.placeholder(np.float32,name='keep_prob')

        # 输出
        self.y_logits,self.y = utils.classifier(self.x,self.keep_prob, name='classifier', image_size=self.image_size)

        # 损失
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logits, labels=self.label))

        #训练准确率
        self.accurancy = utils.get_accurancy(self.y, self.label,name='accurancy')

        # 训练参数
        self.params = []
        for v in tf.trainable_variables():
            self.params.append(v)

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        # logs
        tf.summary.scalar('loss', self.loss)
        for v in tf.trainable_variables():
            tf.summary.histogram(v.name, v)

        self.merged_summary_op = tf.summary.merge_all()

    def train(self, LR=2e-4, B1=0.5, B2=0.999, epoch_size=1000,fromckpt=False,auto_validate=False):
        self.train_op = tf.train.AdamOptimizer(LR, B1, B2).minimize(self.loss, var_list=self.params)
        start_time = time.time()
        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            #是否从断点开始训练
            if fromckpt:
                self.restore()
            if not os.path.exists("logs"):
                os.makedirs("logs")
            self.summary_writer = tf.summary.FileWriter(os.getcwd() + '/logs', graph=sess.graph)
            for epoch in range(epoch_size):
                batch_number = self.datasize // self.batch_size
                for step in range(batch_number):
                    step_start_time = time.time()
                    x_batch, label_batch = data.get_batch(step, self.batch_size,mode = "train")
                    step_io_end_time = time.time()
                    op_list = [self.train_op, self.y_logits, self.loss, self.accurancy,self.merged_summary_op]

                    _, y_logits_out, loss_out, accurancy, summary_str = sess.run(op_list,
                                                               feed_dict={self.x: x_batch, self.label: label_batch , self.keep_prob: 0.5})
                    step_end_time = time.time()
                    print("epoch:[%s/%s]  step:[%s/%s]  loss: %.4f  accurancy: %.4f  steptimeuse: %.4f  io_time: %.4f  totaltime: %4.4f"
                          % (epoch, epoch_size, step, batch_number, loss_out, accurancy, step_end_time-step_start_time,step_io_end_time-step_start_time ,step_end_time - start_time))

                if not os.path.exists("models"):
                    os.makedirs("models")
                self.saver.save(sess, "models" + '/model-' + str(epoch) + '.ckpt')

                print("Saved Model")

                #训练过程中自动调用验证集验证
                if auto_validate and epoch%2 == 0:
                    self.validate_in_tarin(sess,datasize=24000, batch_size=800)


    def restore(self):
        try:
            self.saver.restore(self.sess, tf.train.latest_checkpoint('./models'))
        except:
            print("Previous weights not found")

    def validate_in_tarin(self,sess,datasize=24000, batch_size=800):
        batch_number = datasize // batch_size
        sum = 0
        print("-----------------------start-validating-----------------------")
        for step in range(batch_number):
            x_batch, label_batch = data.get_batch(step, batch_size, mode = "validate")
            op_list = [self.loss, self.accurancy]
            loss, val_accurancy = sess.run(op_list,feed_dict={self.x:x_batch,self.label:label_batch,self.keep_prob: 1.0})
            print("batch: [%s/%s] loss: %.4f   val_accurancy: %.4f"%(step,batch_number,loss,val_accurancy))
            sum += val_accurancy
        avg_accurancy = sum/batch_number
        print("average validation accurancy : %.4f"%avg_accurancy)
        print("-----------------------end-validating-------------------------")

    def validate(self,datasize = 24000, batch_size=128):
        with self.sess as sess:
            self.restore()
            batch_number = datasize // batch_size
            sum = 0
            for step in range(batch_number):
                x_batch, label_batch = data.get_batch(step, batch_size, mode = "validate")
                op_list = [self.loss, self.accurancy]
                loss, val_accurancy = sess.run(op_list,feed_dict={self.x:x_batch,self.label:label_batch,self.keep_prob: 1.0})
                print("batch: [%s/%s] loss: %.4f   val_accurancy: %.4f"%(step,batch_number,loss,val_accurancy))
                sum += val_accurancy
            avg_accurancy = sum/batch_number
            print("average validation accurancy : %.4f"%avg_accurancy)

    def test(self, datasize=4838, batch_size=1000):
        with self.sess as sess:
            self.restore()
            batch_number = datasize // batch_size +1
            for step in range(batch_number):
                x_batch, _ = data.get_batch(step, batch_size, mode = "test")
                op_list = [self.y,]
                y_out = sess.run(op_list, feed_dict={self.x: x_batch})
                data.save_output(y_out[0].tolist())
