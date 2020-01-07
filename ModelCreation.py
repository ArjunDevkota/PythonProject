def modelpredict(l):
    #Importing Library
    import pandas as pd
    import numpy as np
0
    #Loading the dataset into dataframe
    df=pd.read_csv('phishcoop.csv')

    #Spliting the training and test data
    from sklearn.model_selection import train_test_split
    X=df.drop(['Result'],axis=1)
    y=df['Result']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05)

    #Train the model
    from sklearn.ensemble import RandomForestClassifier
    rfc=RandomForestClassifier(n_estimators=150)
    rfc.fit(X_train,y_train)


    #PredictionPart
    predict=rfc.predict(l)
    if predict==1:
        return "Phishing Website"
    else:
        return "Not-Phishing Website"



