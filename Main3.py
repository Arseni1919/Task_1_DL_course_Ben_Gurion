import os
import cv2
import pickle
import numpy as np
import skimage
from skimage.feature import hog
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.svm import LinearSVC, SVC
import sklearn
import matplotlib.pyplot as plt
import random
import time
from pprint import pprint

# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# INPUT
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
data_path = 'Data'  # [!]: Create please 'Data' directory before
cache_path = 'cache'  # [!]: Create please 'cache' directory before

# for test
class_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


# for train
train_indices = list(range(11))

# if you want to run on <<class_indices>> set TEST_MODE to True
TEST_MODE = True

# cache some of information
NEED_TO_USE_CACHE = False
# those printings help to understand the flow of process. Set always to True
NEED_PRINT_PROCESS = True

# if you want to make some tuning set TUNING to true and TEST_MODE to False, and it will run on <<train_indices>>
TUNING = False
name = 'tuning_file_name_for_graph'

'''
THE MAIN LOOP IS ALL THE WAY DOWN OF THIS PYTHON FILE, RIGHT AFTER 'FUNCTIONS' SECTOR
'''
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# PARAMETERS:
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --


np.random.seed(0)  # Seed the random number generator

params = {'Split': {}, 'Prepare': {}, 'Train': {}, 'Summary': {}, 'Report': {}, 'Data': {}, 'Cache': {}}

params['Split']['ratio'] = 20

params['Prepare']['spatial_cell_size'] = (8, 8)
params['Prepare']['orientations'] = 10
params['Prepare']['cells_per_block'] = (3, 3)

params['Train']['C'] = 1

# params['Train']['kernel'] = 'linear'
# params['Train']['kernel'] = 'rbf'
params['Train']['kernel'] = 'poly'

params['Train']['degree'] = 2
params['Train']['gamma'] = 0.01
params['Train']['class_names'] = []

params['Summary']['error_images_per_class'] = 2

params['Report'][''] = []

params['Data']['path'] = data_path
params['Data']['S'] = 125
params['Data']['N_classes'] = 10
params['Data']['max_N_images_per_class'] = 40  # 20 - training, 20 - test

params['Cache']['CachePath'] = cache_path


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# FUNCTIONS:
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
def show_data(times, images, labels=None):
    '''
    This function helps to debug functions that are working with images. It shows 'times' first images.
    :param times: How many images to show
    :param images: list of images
    :param labels: show images with their labels
    :return:
    '''
    if labels is None:
        for ind in range(times):
            img = images[ind]
            plt.imshow(img)
            plt.show()
    else:
        my_list = list(zip(images, labels))
        for ind in range(times):
            img, label = my_list[ind]
            plt.title(label)
            plt.imshow(img)
            plt.show()


def get_parameters():
    '''
    Preprocessing of dict of parameters
    :return: dict of parameters
    '''
    # pprint(params)
    return params


def create_image(path_to_pic, S):
    '''
    cv2 preprocessing
    :param path_to_pic: path to picture
    :param S: S parameter
    :return: ndarray object
    '''
    img = cv2.imread(path_to_pic)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (S, S))
    return img


def create_saved_data(cache_path, name, data_array, labels):
    '''
    Cache created data.
    :param cache_path: path to cache directory.
    :param name: Name of new file.
    :param data_array: data to save.
    :param labels: labels of data.
    :return: tuple of data and it's labels.
    '''
    file_Name = '%s/%s' % (cache_path, name)
    data_ndarray = np.asarray(data_array)
    saved_data = (data_ndarray, labels)

    with open(file_Name, 'wb') as fileObject:
        pickle.dump(saved_data, fileObject)

    return saved_data


def prepare_data_from(indices, params, name):
    '''
    Transforming data from images to ndarrays.
    :param indices: indexes of classes we want to work with.
    :param params: dict of parameters.
    :param name: name of cached file if exist.
    :return: transformed set of data.
    '''
    cache_path = params['Cache']['CachePath']
    data_directory = params['Data']['path']
    S = params['Data']['S']
    max_N_images_per_class = params['Data']['max_N_images_per_class']
    dirs = get_ordered_list(os.listdir(data_directory))
    data_array = []
    labels = []
    print_statement('Starting to load the data...')

    for name_of_class, index_of_class in zip(dirs, range(len(dirs))):
        if name_of_class != '.DS_Store':
            if index_of_class in indices:

                path_into_class = '%s/%s' % (data_directory, name_of_class)
                dirs_of_class = get_ordered_list(os.listdir(path_into_class))
                for pic, index_of_pic in zip(dirs_of_class, range(len(dirs_of_class))):
                    data_array.append(create_image('%s/%s' % (path_into_class, pic), S))
                    labels.append(name_of_class)
                    if index_of_pic + 1 == max_N_images_per_class:
                        break

    return create_saved_data(cache_path, name, data_array, labels)


def get_from_chache(cache_path, name):
    '''
    Export from cache directory.
    :param cache_path: path to cache directory.
    :param name: Name of new file.
    :return: loaded data
    '''
    file_Name = '%s/%s' % (cache_path, name)
    with open(file_Name, 'rb') as fileObject:
        loaded_data = pickle.load(fileObject)
        print('[data_loader]: Data already saved.')
        return loaded_data[0], loaded_data[1]


def data_loader(params, indices):
    '''
    Loads the data from folders.
    :param params: parameters
    :param indices: indexes of classes we want to work with.
    :return: data
    '''
    # create unique name
    name = ''
    cache_path = params['Cache']['CachePath']
    for i in indices:
        name = '%s[%s]' % (name, i)

    if NEED_TO_USE_CACHE:
        list_dir = os.listdir(cache_path)
        if name in list_dir:
            return get_from_chache(cache_path, name)
    return prepare_data_from(indices, params, name)


def train_test_split(data, labels, params):
    '''
    Splits the data to: train_data, test_data, train_labels, test_labels
    :param data: data
    :param labels: labels
    :param params: parameters
    :return: tuple of train_data, test_data, train_labels, test_labels
    '''
    ratio = params['Split']['ratio']
    train_data, test_data, train_labels, test_labels = [], [], [], []
    print_statement('Starting to split the data...')
    for label in get_ordered_list(labels):

        counter_for_train = 0
        for pic, each_label in zip(data, labels):
            if each_label == label:
                if counter_for_train < ratio:
                    train_data.append(pic)
                    train_labels.append(label)
                else:
                    test_data.append(pic)
                    test_labels.append(label)
                counter_for_train += 1

    params['Train']['class_names'] = get_ordered_list(train_labels)
    return train_data, test_data, train_labels, test_labels


def prepare(some_data, params):
    '''
    HOG transformation.
    :param some_data: train or test data in our case
    :param params: parameters
    :return: HOG representation of images and actual pictures of the result (we can display them if we want)
    '''
    orientations = params['Prepare']['orientations']
    spatial_cell_size = params['Prepare']['spatial_cell_size']
    cells_per_block = params['Prepare']['cells_per_block']
    images = []
    hog_images = []
    print_statement('HOG transformation...')
    for pic in some_data:
        hog_image, image = hog(pic,
                               orientations=orientations,
                               pixels_per_cell=spatial_cell_size,
                               cells_per_block=cells_per_block,
                               block_norm='L2',
                               visualize=True)

        hog_images.append(hog_image)
        images.append(image)

    return hog_images, images


def tuning_linear(train_data_rep, train_labels, params):
    '''
    K-folds tuning.
    :param train_data_rep: matrix of train data
    :param train_labels: list of train labels
    :param params: parameters
    :return: model
    '''
    C = params['Train']['C']

    train_data_rep_np = np.asarray(train_data_rep)
    train_labels_np = np.asarray(train_labels)

    m_classes_svm = LinearSVC(C=C, multi_class='ovr')
    cross_val_scores = cross_val_score(m_classes_svm, train_data_rep_np, train_labels_np, cv=4)
    return m_classes_svm, cross_val_scores.mean()


def tuning_non_linear(train_data_rep, train_labels, params):
    '''
    K-folds tuning.
    :param train_data_rep: matrix of train data
    :param train_labels: list of train labels
    :param params: parameters
    :return: returns list of models
    '''
    cross_val_scores = []
    train_data_rep_np = np.asarray(train_data_rep)
    train_labels_np = np.asarray(train_labels)
    kf = StratifiedKFold(n_splits=4)

    for train_index, test_index in kf.split(train_data_rep_np, train_labels_np):
        X_train, X_test = train_data_rep_np[train_index], train_data_rep_np[test_index]
        y_train, y_test = train_labels_np[train_index], train_labels_np[test_index]

        m_classes_svm = create_non_linear_classifier(X_train, y_train, params)

        predictions, _ = m_classes_svm_predict(m_classes_svm, X_test, params)
        cross_val_scores.append(sklearn.metrics.accuracy_score(y_test, predictions))
    accuracy = np.asarray(cross_val_scores).mean()

    return create_non_linear_classifier(train_data_rep, train_labels, params), accuracy


def get_ordered_list(curr_list):
    '''
    :param curr_list: some list
    :return: ordered set of items of a given list
    '''
    my_list = list(set(curr_list))
    my_list.sort(key=lambda v: v.upper())
    return my_list


def create_non_linear_classifier(train_data_rep, train_labels, params):
    '''
    Help function of tuning_non_linear(). For non linear cases.
    :return: list of models
    '''
    kernel = params['Train']['kernel']
    C = params['Train']['C']
    gamma = params['Train']['gamma']
    degree = params['Train']['degree']
    m_classes_svm = []

    for label in get_ordered_list(train_labels):
        temp_binary_labels = []
        for pic, curr_label in zip(train_data_rep, train_labels):
            if label == curr_label:
                temp_binary_labels.append(1)
            else:
                temp_binary_labels.append(-1)
        m_classes_svm.append(SVC(kernel=kernel, gamma=gamma, degree=degree, C=C))
        m_classes_svm[-1].fit(train_data_rep, temp_binary_labels)
    return m_classes_svm


def m_classes_svm_train(train_data_rep, train_labels, params):
    '''
    Create the classifier or list of classifiers.
    :param train_data_rep: matrix of Train data
    :param train_labels: list of Train labels
    :param params: parameters
    :return: the classifier or list of classifiers
    '''
    kernel = params['Train']['kernel']
    C = params['Train']['C']
    print_statement('Training...')
    train_data_rep_np = np.asarray(train_data_rep)
    train_labels_np = np.asarray(train_labels)

    if kernel == 'linear':
        if TUNING:
            return tuning_linear(train_data_rep, train_labels, params)
        return LinearSVC(C=C, multi_class='ovr').fit(train_data_rep_np, train_labels_np), 0
    if TUNING:
        return tuning_non_linear(train_data_rep_np, train_labels_np, params)
    return create_non_linear_classifier(train_data_rep_np, train_labels_np, params), 0


def m_classes_svm_predict(models, test_data_rep, params):
    '''
    Create predictions.
    :param models: The SVM model or list of SVM models.
    :param test_data_rep: Test data.
    :param params: parameters.
    :return: list of predictions, scores
    '''
    kernel = params['Train']['kernel']
    class_names = params['Train']['class_names']
    print_statement('Predicting...')
    if kernel == 'linear':
        predictions = models.predict(test_data_rep)
        scores = models.decision_function(test_data_rep)
        return predictions, scores
    else:
        scores = np.empty((len(test_data_rep), len(models)))
        for model, index_of_model in zip(models, range(len(models))):
            scores[:, index_of_model] = model.decision_function(test_data_rep)
        max_score_indxs = np.apply_along_axis(np.argmax, 1, scores)
        predictions = []
        for i in max_score_indxs:
            predictions.append(class_names[i])

        return predictions, scores


def get_wrongly_classified_pics(label, index_of_label, test_labels, scores, num_of_pics):
    '''
    Help function for get_indices_of_the_largest_error_images(). Return wrongly classified pics.
    '''
    array_of_pics = []
    for pic_label, pic, index_of_pic in zip(test_labels, scores, range(num_of_pics)):
        if pic_label == label:
            if np.argmax(pic) != index_of_label:
                array_of_pics.append(index_of_pic)
    return array_of_pics


def get_sorted_indexes_of_wrong_pics(array_of_pics, index_of_label):
    '''
    Help function for get_indices_of_the_largest_error_images(). Returns sorted list.
    '''
    array_of_margins = []
    for index_of_pic in array_of_pics:
        pic = scores[index_of_pic]
        array_of_margins.append(pic[index_of_label] - np.max(pic))

    return [index_of_pic for _, index_of_pic in sorted(zip(array_of_margins, array_of_pics),
                                                       key=lambda pair: pair[0], reverse=True)]


def get_indices_of_the_largest_error_images(scores, test_labels, params):
    '''
    :param scores: scores of images per each class
    :param test_labels: labels of test set
    :param params: parameters
    :return: dict_of_error_pics[label] = set of two error pics with maximum margin
    '''
    error_images_per_class = params['Summary']['error_images_per_class']
    distinct_label_names = get_ordered_list(test_labels)
    num_of_pics = len(test_labels)
    dict_of_error_pics = {}

    for label, index_of_label in zip(distinct_label_names, range(len(distinct_label_names))):
        dict_of_error_pics[label] = []

        array_of_pics = get_wrongly_classified_pics(label, index_of_label, test_labels, scores, num_of_pics)

        sorted_pics = get_sorted_indexes_of_wrong_pics(array_of_pics, index_of_label)

        for index_of_pic in sorted_pics:
            if len(dict_of_error_pics[label]) == error_images_per_class:
                break
            dict_of_error_pics[label].append(index_of_pic)

    return dict_of_error_pics


def evaluate_model(test_labels, params, scores, predictions):
    '''
    Preparing information to show.
    :param test_labels: labels of test set
    :param params: parameters
    :param scores: scores of images per each class
    :param predictions: predicted labels
    :return: mean_accuracy, disp_confusion_matrix, indices_of_the_largest_error_images
    '''
    print_statement('Evaluation of the model:')
    display_labels = get_ordered_list(test_labels)
    mean_accuracy = sklearn.metrics.accuracy_score(test_labels, predictions)
    confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, predictions)
    disp_confusion_matrix = ConfusionMatrixDisplay(confusion_matrix, display_labels).plot()
    indices_of_the_largest_error_images = get_indices_of_the_largest_error_images(scores, test_labels, params)
    return mean_accuracy, disp_confusion_matrix, indices_of_the_largest_error_images


def report_results(summary, params, test_data):
    '''
    Reports beautiful graphs and prints the necessary outputs.
    :param summary: mean_accuracy, disp_confusion_matrix, indices_of_the_largest_error_images
    :param params: parameters
    :param test_data: data
    '''
    mean_accuracy, disp_confusion_matrix, indices_of_the_largest_error_images = summary
    print('The test error: ', (1 - mean_accuracy))
    print('The mean accuracy: ', mean_accuracy)
    print('Confusion matrix: \n', disp_confusion_matrix.confusion_matrix)
    # pprint(indices_of_the_largest_error_images)
    plt.xticks(rotation=45)
    plt.title('Confusion Matrix')
    num_of_classes = len(list(indices_of_the_largest_error_images.keys()))
    error_images_per_class = params['Summary']['error_images_per_class']

    fig, ax = plt.subplots(nrows=num_of_classes, ncols=(error_images_per_class + 1), figsize=(5.5, 5.5))
    for (label, indecies), class_index in zip(indices_of_the_largest_error_images.items(), range(num_of_classes)):
        for i, i_order in zip(indecies, range(len(indecies))):
            ax[class_index][i_order].imshow(test_data[i])
        ax[class_index][error_images_per_class].text(0, 0.5, '<-- %s' % label,
                                                     horizontalalignment='center',
                                                     verticalalignment='center', fontsize=10, color='red', )
        for curr in ax[class_index]:
            curr.axis('off')
    fig.canvas.set_window_title('Error Images')
    plt.suptitle('Error Images')
    plt.show()


def plot_graph(name, legend_labels='', x_label='', kernel=''):
    '''
    Plots beautiful graph.
    '''
    plt.style.use('bmh')
    with open(name, 'rb') as fileObject:
        to_save = pickle.load(fileObject)

        x_labels = to_save['x_labels']
        y = to_save['y']
        x = to_save['x']

        x_j_labels, x_i_labels = x_labels
        for j in range(len(x)):
            plt.plot(x[j], y[j], label='%s=%s' % (legend_labels, x_j_labels[j]))
            plt.xticks(x[j], x_i_labels)

        plt.xlabel('%s Value' % x_label)
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy vs %s & %s - %s SVM Model' % (legend_labels, x_label, kernel))
        # plt.legend(loc='lower right')
        plt.legend()
        plt.show()


def save_and_plot(to_save, name):  # {'x_labels': (x_j_labels, x_i_labels), 'x': x, 'y': y}
    '''
    Help function to save and plot the results.
    :param to_save: dict of graph information
    :param name: name of file to write to
    '''
    with open(name, 'wb') as fileObject:
        pickle.dump(to_save, fileObject)
    plot_graph(name, legend_labels='', x_label='', kernel='')


def print_statement(statement):
    '''
    Beautiful printing along the process.
    :param statement: string to print
    :return:
    '''
    if NEED_PRINT_PROCESS:
        print('[ARSENI & BAR]: %s' % statement)


# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --
# MAIN:
# --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --


if __name__ == '__main__':
    # ********** Settings: ********** #
    counter, x, y = 0, [], []

    x_j_labels = [50, 100, 125, 150]  # S
    x_i_labels = [10, 20, 30, 40, 50]  # orientations

    classes = class_indices if TEST_MODE else train_indices
    print('The classes are: ', classes)
    TUNING = False if TEST_MODE else TUNING

    if TUNING:
        for j in x_j_labels:
            indx = 0
            x.append([])
            y.append([])
            for i in x_i_labels:
                x[-1].append(indx)
                indx += 1
                params = get_parameters()
                # ************************* What we change during tuning: ************************* #
                params['Data']['S'] = j
                params['Prepare']['orientations'] = i
                # ********************************************************************************* #
                data, labels = data_loader(params, train_indices)
                train_data, _, train_labels, _ = train_test_split(data, labels, params)
                train_data_rep, hog_images = prepare(train_data, params)
                # show_data(3, hog_images, train_labels)
                _, accuracy = m_classes_svm_train(train_data_rep, train_labels, params)
                y[-1].append(accuracy)
                counter += 1
                print(counter, '/', (len(x_i_labels) * len(x_j_labels)),
                      ' Values ', j, ' and ', i, ": get accuracy of ", accuracy)

        save_and_plot({'x_labels': (x_j_labels, x_i_labels), 'x': x, 'y': y}, name)

    data, labels = data_loader(params, classes)
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, params)
    train_data_rep, _ = prepare(train_data, params)
    test_data_rep, _ = prepare(test_data, params)

    models, _ = m_classes_svm_train(train_data_rep, train_labels, params)
    predictions, scores = m_classes_svm_predict(models, test_data_rep, params)

    summary = evaluate_model(test_labels, params, scores, predictions)
    report_results(summary, params, test_data)

'''
- take 10 first classes (.DS_Store)
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
PARAMETERS:

HOG:
- 'celss_size' spatial cell size
- 'num_orient' number of orientation bins

SVM:
- 'C' the SVM tradeoff parameter 
- 'gamma' when using an RBF kernel
- 'degree' the polynomial degree when using a polynomial kernel
------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------

'''