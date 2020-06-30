import psycopg2
import tensorflow as tf
import logging
from src.config import predict_categories


class Db2Tfrecord:
    def query_db(self):
        raise NotImplementedError

    def format_data(self):
        raise NotImplementedError

    def write2tfrecord(self):
        raise NotImplementedError


class PostgreSQL2Tfrecord(Db2Tfrecord):
    def __init__(self):
        self.config = 1

    def query_db(self):
        try:
            conn = psycopg2.connect(
                database="database",
                user="user",
                password="password",
                host="localhost",
                port="5432")
            cur = conn.cursor()

            cur.execute("select * from IRIS")
            rows = cur.fetchall()
            data = {}
            data["sepal_length"] = []
            data["sepal_width"] = []
            data["petal_length"] = []
            data["petal_width"] = []
            data["variety"] = []

            for row in rows:
                data["sepal_length"].append(row[1])
                data["sepal_width"].append(row[2])
                data["petal_length"].append(row[3])
                data["petal_width"].append(row[4])
                data["variety"].append(row[5])
            conn.commit()
            conn.close()

        except Exception as e:
            logging.error("Can not query from db: {}".format(e))
            return 0
        return data

    def format_data(self, data):
        data["variety"] = list(
            map(lambda x: predict_categories.index(x), data["variety"]))
        return data

    def write2tfrecord(self, data, filename):
        with tf.io.TFRecordWriter(filename) as writer:
            print(data.keys())
            for i in range(len(data[list(data.keys())[0]])):
                feature = {
                    "sepal_length": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[data["sepal_length"][i]])),
                    "sepal_width": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[data["sepal_width"][i]])),
                    "petal_length": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[data["petal_length"][i]])),
                    "petal_width": tf.train.Feature(
                        float_list=tf.train.FloatList(
                            value=[data["petal_width"][i]])),
                    "variety": tf.train.Feature(
                        int64_list=tf.train.Int64List(
                            value=[data["variety"][i]]))
                }
                example_proto = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                example = example_proto.SerializeToString()
                writer.write(example)


class Pipeline:
    def __init__(self, filenames, config):

        self.epochs = config["epochs"]
        full_dataset = tf.data.TFRecordDataset(filenames)
        data_size = 0
        for row in full_dataset:
            data_size += 1
        train_size = int(0.7 * data_size)
        val_size = int(0.15 * data_size)
        test_size = int(0.15 * data_size)
        full_dataset = full_dataset.shuffle()
        self.train_dataset = full_dataset.take(train_size)
        self.test_dataset = full_dataset.skip(train_size)
        self.val_dataset = self.test_dataset.skip(test_size)
        self.test_dataset = self.test_dataset.take(test_size)

    def get_train_data(self):
        return self.train_dataset.repeat(self.epochs)

    def get_val_data(self):
        return self.val_dataset.repeat(self.epochs)

    def get_test_data(self):
        return self.test_dataset.repeat(self.epochs)
