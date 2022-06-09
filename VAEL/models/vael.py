from utils import *


class config:
    batchsize = 64
    l_f = 249
    pre_epoch = 20
    ft_epoch = 100
    lr = 0.0001
    S = 256
    M = 512
    train = True
    SaveModlFile_1 = './save/model_v5_b1.h5'
    SaveModlFile_2 = './save/model_v5_b2.h5'


if __name__ == '__main__':
    gpu_setting()
    set_seed()

    # train CNN1
    x_train_1, x_val_1, y_train_1, y_val_1, x_test_1, y_test = get_data(config.S, reshape=False)
    if config.train:
        train_model(x_train_1, x_val_1, y_train_1, y_val_1, config, config.SaveModlFile_1)

    prediction_test_cnn1 = predict(x_test_1, config.SaveModlFile_1)
    accuracy, precision, recall, f1_score, AUC = evaluate(y_test, prediction_test_cnn1)
    print('Performance for CNN1:')
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-score: ', f1_score)
    print('AUC: ', AUC)
    print('#########################')

    # train CNN2
    x_train_2 = resize(x_train_1, (config.M, config.M))
    x_test_2 = resize(x_test_1, (config.M, config.M))
    x_val_2 = resize(x_val_1, (config.M, config.M))

    if config.train:
        train_model(x_train_2, x_val_2, y_train_1, y_val_1, config, config.SaveModlFile_2)

    prediction_test_cnn2 = predict(x_test_2, config.SaveModlFile_2)
    accuracy, precision, recall, f1_score, AUC = evaluate(y_test, prediction_test_cnn2)
    print('Performance for CNN2:')
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-score: ', f1_score)
    print('AUC: ', AUC)
    print('#########################')

    # ensemble
    prediction_test_vael = ensemble(x_test_1, x_test_2, config)
    accuracy, precision, recall, f1_score, AUC = evaluate(y_test, prediction_test_vael)
    print('Performance for VAEL:')
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-score: ', f1_score)
    print('AUC: ', AUC)
