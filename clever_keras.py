import time
import numpy
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from getfeature import return_data


seed = 10
numpy.random.seed(None)

X,Y = return_data()

#encoder = LabelEncoder()
#encoder.fit(Y)
#encoded_Y = encoder.transform(Y)
#print(encoded_Y)
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=10, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model






def create_larger():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=10, init='normal', activation='sigmoid'))
    from keras.layers import Dropout
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    return model

def stupid_keras():
    start_time = time.time()
    numpy.random.seed(seed)
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasClassifier(build_fn=create_larger, nb_epoch=100, batch_size=5, verbose=0)))
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(pipeline, X, Y,scoring='f1_weighted', cv=kfold)
    print("KERAS: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    print(results)


if __name__ == '__main__':
    
    stupid_keras()

    clf = svm.SVC(kernel='linear', C=1)
    k = KFold(n_splits=10, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X,Y, scoring='f1_weighted',cv=10)
    print("SVM: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))
    print(scores)


    
    estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, X, Y, cv=kfold) 
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

   
