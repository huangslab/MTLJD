# coding:utf-8
from MMOE import MMoE
import keras
from keras import Input, Model
from keras.layers import Dense, Concatenate, Lambda, Dropout, Add, Multiply, AveragePooling1D
import sklearn.metrics as metrics
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.initializers import VarianceScaling
from keras import Model

def M_MMOE_with_pheno(X_train, y_train, X_train_pheno, X1_test, y1_test, X1_test_pheno, X2_test, y2_test,
                           X2_test_pheno, X3_test, y3_test, X3_test_pheno, cluster_indices, selected_feature_index,
                           bdry, mmoe_units, experts_num, tasks_num):
    K.clear_session()

    train_data = []
    ASD_test = []
    ADHD_test = []
    COBRE_test = []
    mmoe_layer_ASD = []
    mmoe_layer_ADHD = []
    mmoe_layer_COBRE = []
    input_para = []

    # MMOE
    for i in range(len(cluster_indices)):
        data = X_train[:, cluster_indices[i]]
        test1 = X1_test[:, cluster_indices[i]]
        test2 = X2_test[:, cluster_indices[i]]
        test3 = X3_test[:, cluster_indices[i]]

        data = data[:, selected_feature_index[i]]
        test1 = test1[:, selected_feature_index[i]]
        test2 = test2[:, selected_feature_index[i]]
        test3 = test3[:, selected_feature_index[i]]

        train_data.append(data)
        ASD_test.append(test1)
        ADHD_test.append(test2)
        COBRE_test.append(test3)

        locals()['main_input' + str(i)] = Input((data.shape[1],), name='main_input' + str(i))
        input_para.append(locals()['main_input' + str(i)])
        locals()['mmoe_layers' + str(i)] = MMoE(mmoe_units, experts_num, tasks_num)(locals()['main_input' + str(i)])
        mmoe_layer_ASD.append(locals()['mmoe_layers' + str(i)][0])
        mmoe_layer_ADHD.append(locals()['mmoe_layers' + str(i)][1])
        mmoe_layer_COBRE.append(locals()['mmoe_layers' + str(i)][2])

    train_data.append(X_train_pheno)
    ASD_test.append(X1_test_pheno)
    ADHD_test.append(X2_test_pheno)
    COBRE_test.append(X3_test_pheno)

    pheno_input = Input((3,), name='phenotype_input')
    input_para.append(pheno_input)

    mmoe_output_ASD = Concatenate()(mmoe_layer_ASD)
    mmoe_output_ADHD = Concatenate()(mmoe_layer_ADHD)
    mmoe_output_COBRE = Concatenate()(mmoe_layer_COBRE)

    mmoe_ASD_with_pheno = Concatenate()([mmoe_output_ASD, pheno_input])
    mmoe_ADHD_with_pheno = Concatenate()([mmoe_output_ADHD, pheno_input])
    mmoe_COBRE_with_pheno = Concatenate()([mmoe_output_COBRE, pheno_input])

    # task specific
    output_layers = []
    mmoe_layers = [mmoe_ASD_with_pheno, mmoe_ADHD_with_pheno, mmoe_COBRE_with_pheno]
    output_info = ['ASD', 'ADHD', 'COBRE']
    # Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = Dense(
            units=64,
            kernel_initializer=VarianceScaling(),
            activation='relu')(task_layer)
        tower_layer = Dropout(0.25)(tower_layer)

        tower_layer = Dense(
            units=10,
            kernel_initializer=VarianceScaling(),
            activation='relu')(tower_layer)

        MMOE_output_layer = Dense(
            units=1,
            name=output_info[index],
            activation='linear')(tower_layer)
        output_layers.append(MMOE_output_layer)
    # build a new model
    model_MMOE = Model(inputs=input_para, outputs=output_layers)

    print(model_MMOE.summary())

    # compile
    model_MMOE.compile(optimizer='Adadelta',
                       loss={'ASD': 'mean_squared_error', 'ADHD': 'mean_squared_error', 'COBRE': 'mean_squared_error'},
                       loss_weights=[1, 1, 1],
                       metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    earlystop = keras.callbacks.EarlyStopping(monitor='val_ASD_accuracy', min_delta=0.0001, patience=5, verbose=1,
                                              mode='auto')
    model_MMOE.fit(x=train_data, y=y_train, epochs=20, batch_size=64, validation_split=0.1, callbacks=[reduce_lr])

    y_MMOE_ASD_predict = model_MMOE.predict(x=ASD_test)
    y_MMOE_ADHD_predict = model_MMOE.predict(x=ADHD_test)
    y_MMOE_COBRE_predict = model_MMOE.predict(x=COBRE_test)

    y_score = []
    y_score.append(y_MMOE_ASD_predict[0])
    y_score.append(y_MMOE_ADHD_predict[1])
    y_score.append(y_MMOE_COBRE_predict[2])

    ASD_pred = [1 if prob >= bdry else 0 for prob in y_MMOE_ASD_predict[0]]
    ADHD_pred = [1 if prob >= bdry else 0 for prob in y_MMOE_ADHD_predict[1]]
    COBRE_pred = [1 if prob >= bdry else 0 for prob in y_MMOE_COBRE_predict[2]]

    ASD_acc = metrics.accuracy_score(y1_test, ASD_pred)
    ADHD_acc = metrics.accuracy_score(y2_test, ADHD_pred)
    COBRE_acc = metrics.accuracy_score(y3_test, COBRE_pred)

    print('ASD mmoe' + str(i) + ' acc: ' + str(ASD_acc))
    print('ADHD mmoe' + str(i) + ' acc: ' + str(ADHD_acc))
    print('COBRE mmoe' + str(i) + ' acc: ' + str(COBRE_acc))

    return ASD_acc, ADHD_acc, COBRE_acc



