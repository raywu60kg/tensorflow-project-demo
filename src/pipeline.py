import psycopg2
import tensorflow as tf
import logging
from src.config import predict_categories, label_name, feature_names


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
        data[label_name] = list(
            map(lambda x: predict_categories.index(x), data["variety"]))
        data[label_name] = tf.keras.utils.to_categorical(data[label_name])
        return data

    def write2tfrecord(self, data, filename):
        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(len(data[list(data.keys())[0]])):
                # variety_feature_list = []
                # for element in data["variety"][i]:
                #     variety_feature_list.append(
                #         tf.train.FloatList(value=[element]))

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
                        float_list=tf.train.FloatList(
                            value=data["variety"][i]))
                }
                example_proto = tf.train.Example(
                    features=tf.train.Features(feature=feature))
                example = example_proto.SerializeToString()
                writer.write(example)


class Pipeline:
    def __init__(self, tfrecords_filenames):
        self.features = {
            "sepal_length": tf.io.FixedLenFeature([], tf.float32),
            "sepal_width": tf.io.FixedLenFeature([], tf.float32),
            "petal_length": tf.io.FixedLenFeature([], tf.float32),
            "petal_width": tf.io.FixedLenFeature([], tf.float32),
            "variety": tf.io.FixedLenFeature((3), tf.float32)
        }

        full_dataset = tf.data.TFRecordDataset(tfrecords_filenames)
        data_size = 0
        for row in full_dataset:
            data_size += 1
        self.data_size = data_size
        train_size = int(0.7 * data_size)
        val_size = int(0.15 * data_size)
        test_size = int(0.15 * data_size)
        self.train_size = train_size
        full_dataset = full_dataset.shuffle(buffer_size=1)
        full_dataset = full_dataset.map(self.parse_data)
        self.train_dataset = full_dataset.take(train_size)
        self.test_dataset = full_dataset.skip(train_size)
        self.val_dataset = self.test_dataset.skip(test_size)
        self.test_dataset = self.test_dataset.take(test_size)

    def parse_data(self, serialized):
        parsed_example = tf.io.parse_example(serialized, self.features)

        inputs = {}
        for key in parsed_example.keys():
            if key in feature_names:
                inputs[key] = parsed_example[key]
        return (inputs, {label_name: parsed_example[label_name]})

    def get_train_data(self, batch_size):
        return self.train_dataset.batch(batch_size)

    def get_val_data(self, batch_size):
        return self.val_dataset.batch(batch_size)

    def get_test_data(self, batch_size):
        return self.test_dataset.batch(batch_size)
