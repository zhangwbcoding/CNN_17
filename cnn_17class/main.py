from model import CNN_17
import tensorflow as tf

# parser = argparse.ArgumentParser(description='CNN_17')
# parser.add_argument('--train', type=bool, default=False, help='train the network')
# args = parser.parse_args()

flags = tf.app.flags
flags.DEFINE_string("mode", "validate", "True for training, False for testing [False]")
flags.DEFINE_boolean("fromckpt",False,"Using ckpt to train the model")
flags.DEFINE_boolean("auto_validate",False,"validating the model while training")
FLAGS = flags.FLAGS


mode = FLAGS.mode
fromckpt = FLAGS.fromckpt
auto_validate = FLAGS.auto_validate
print("buliding Model")

if mode == "train":
    network = CNN_17()
    print("Begining Training..")
    network.train(fromckpt=fromckpt,auto_validate=auto_validate)
elif mode == "validate":
    network = CNN_17()
    print("Begining Validating..")
    network.validate(datasize=24000,batch_size=1000)
elif mode == "test":
    network = CNN_17()
    print("Begining testing..")
    network.test(datasize=4838, batch_size=1000)