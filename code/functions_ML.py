from modules import *
def getSlope(X,Y):
    assert len(X) == len(Y)
    N= len(X)
    
    meanx =X.mean()
    meany =Y.mean()
    
    xy = [X[i]*Y[i] for i in range(N)]
    xsquare = [X[i]**2 for i in range(N)]
    
    num = sum(xy)-N*meanx*meany
    denom= sum(xsquare)-N*((meanx)**2)
    
    return meanx,meany,num/denom


def Rsquared(dfT,crime,othercrime):
    X = dfT[str(crime)]
    Y = dfT[str(othercrime)]
    
    N= len(X)
    meanx,meany,a = getSlope(X,Y) # a is the slope
    b = meany-a*meanx # intercept
    f = a*X + b
    
    #total sum of squares
    SStot = sum([(yi-meany)**2 for yi in Y])
    
    # residual sum of squares
    SSres = sum([(yi-(a*X[i]+b))**2 for i,yi in enumerate(Y)])
    
    Rsquared = 1-(SSres/SStot)
    
    return Rsquared


def model_assess_old(model, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    #print(confusion_matrix(y_test, preds))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')


def model_assess(model, X_train, X_test, y_train, y_test,
                 title = "Default", 
                 multiClass= False, 
                 averageMulti="weighted",
                 print_stat = True):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if multiClass== False: 
        accuracy = round(accuracy_score(y_test, preds), 5)
        precision =round(precision_score(y_test, preds,pos_label=1), 5)
        recall = round(recall_score(y_test, preds,average="binary", pos_label=1), 5)
        f1 = round(f1_score(y_test, preds), 5)
    
    else: 
        accuracy = round(accuracy_score(y_test, preds), 5)
        precision =round(precision_score(y_test, preds,pos_label=1,average=averageMulti,zero_division=1), 5)
        recall = round(recall_score(y_test, preds,average=averageMulti, pos_label=1,labels=np.unique(preds)), 5)
        f1 = round(f1_score(y_test, preds,average=averageMulti,labels=np.unique(preds)), 5)
    
    CM = confusion_matrix(y_test, preds)
    
    metrics = {
        "accuracy":accuracy, 
        "precision":precision ,
        "recall":recall, 
        "f1":f1
        }

    if print_stat == True:
        print(f"\n{title}\n{'-'*10}")
        #print(f"{CM}")
        print(f"{accuracy=}")
        print (f"{precision=}")
        print (f"{recall=}")
        print (f"{f1=}\n")
    
        sns.heatmap(pd.DataFrame(CM), annot=True,fmt='d',annot_kws={"size": 16},linewidths=.5)
        plt.show()
    return preds, metrics

def getPrecision(predictions,y_test, print_stat = True):
    # Precision
    TP = sum([1 if p == y_test[y_test.index[i]] else 0 for i,p in enumerate(predictions)])
    errors = len(predictions)-TP 
    precision = TP/len(predictions)
    if print_stat == True:
        print (f"{errors=}\n{precision=}")
    return precision


def runClassifierTreeAndForest(df, features, target,predType,
                               
                               test_size=0.33, 
                               random_state=42,
                               min_samples_split=20, 
                               
                               n_estimators=200, 
                               
                               print_stat = True
                                ):
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, 
                                                        random_state=random_state
                                                       )
    if print_stat == True :
        print (f"\n{predType=}")
    if predType == "tree":
        ##TREE
        dt = DecisionTreeClassifier(min_samples_split=min_samples_split, 
                                    random_state=random_state
                                   )
        tree =dt.fit(X_train, y_train)
        # Use the tree's predict method on the test data
        predictions = tree.predict(X_test)
        #get precision
        precision = getPrecision(predictions,y_test,print_stat = print_stat)
        
        return tree,predictions,precision
    
    
    elif predType == "forest":
        ## FOREST
        df = RandomForestClassifier(n_estimators=n_estimators, 
                                    random_state=random_state,
                                    criterion='gini'
                                    )
        forest =df.fit(X_train, y_train)
        
        predictions = forest.predict(X_test)
        #get precision
        precision = getPrecision(predictions,y_test,print_stat = print_stat)
        
        return forest,predictions,precision
    
    else:
        print ("only available tree or forest")

def runClassifierTreeAndForest2(df, X, y,predType,
                               
                               test_size=0.33, 
                               random_state=42,
                               min_samples_split=20, 
                               
                               n_estimators=200, 
                               
                               print_stat = True
                                ):

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, 
                                                        random_state=random_state
                                                       )
    if print_stat == True:                                            
        print (f"\n{predType=}")
    if predType == "tree":
        ##TREE
        dt = DecisionTreeClassifier(min_samples_split=min_samples_split, 
                                    random_state=random_state
                                   )
        tree =dt.fit(X_train, y_train)
        # Use the tree's predict method on the test data
        predictions = tree.predict(X_test)
        #get precision
        precision = getPrecision(predictions,y_test,print_stat = print_stat)
        
        return tree,predictions,precision
    
    
    elif predType == "forest":
        ## FOREST
        df = RandomForestClassifier(n_estimators=n_estimators, 
                                    random_state=random_state,
                                    criterion='gini')
        forest =df.fit(X_train, y_train)
        
        predictions = forest.predict(X_test)
        #get precision
        precision = getPrecision(predictions,y_test,print_stat = print_stat)
        
        return forest,predictions,precision
    
    else:
        print ("only available tree or forest")


def trainAndCompare(X_train, X_test, y_train, y_test,classifiers,
                    averageMulti = "weighted",
                    print_stat = False, multiClass = False):
    dict_metrics = {}
    
    if "Naive Bayes" in classifiers:
        # Naive Bayes
        nb = GaussianNB()
        preds, metrics = model_assess(nb, X_train, X_test, y_train, y_test, "Naive Bayes",
                                    averageMulti=averageMulti,
                                    print_stat = print_stat,
                                    multiClass = multiClass)
        dict_metrics["Naive Bayes"] = metrics


    if "Decision trees" in classifiers:
        # Decission trees
        tree = DecisionTreeClassifier(min_samples_split=20)
        preds, metrics = model_assess(tree, X_train, X_test, y_train, y_test, "Decision trees", 
                                    averageMulti=averageMulti,
                                    print_stat = print_stat,
                                    multiClass = multiClass)
        dict_metrics["Decision trees"] = metrics

    if "Random Forest" in classifiers:
        # Random Forest
        rforest = RandomForestClassifier(n_estimators=200,criterion='gini', max_depth=5)
        preds, metrics = model_assess(rforest, X_train, X_test, y_train, y_test, "Random Forest",
                                    averageMulti=averageMulti,
                                    print_stat = print_stat,
                                    multiClass = multiClass)
        dict_metrics["Random Forest"] = metrics

    if "Logistic Regression" in classifiers:
        # Logistic Regression
        lg = LogisticRegression(random_state=42,
                                #solver='lbfgs',
                                solver= 'saga' ,
                                max_iter=200,
                                multi_class='multinomial')
        preds, metrics = model_assess(lg, X_train, X_test, y_train, y_test, "Logistic Regression",
                                    averageMulti=averageMulti,
                                    print_stat = print_stat,
                                    multiClass = multiClass)
        dict_metrics["Logistic Regression"] = metrics

    if "CGB" in classifiers:
        # Cross Gradient Booster
        xgb = XGBClassifier(n_estimators=100,eval_metric ='mlogloss', 
                            learning_rate=0.05,use_label_encoder=False,
                            verbosity=0 #silencing warnings
                        )
        preds, metrics = model_assess(xgb, X_train, X_test, y_train, y_test, "Cross Gradient Booster", 
                                    averageMulti=averageMulti,
                                    print_stat = print_stat,
                                    multiClass = multiClass)
        dict_metrics["CGB"] = metrics

    if "CGBRF" in classifiers:
        # Cross Gradient Booster (Random Forest)
        xgbrf = XGBRFClassifier(use_label_encoder=False,
                            verbosity=0
                            )
        preds, metrics = model_assess(xgbrf, X_train, X_test, y_train, y_test, "Cross Gradient Booster (Random Forest)", 
                                      averageMulti=averageMulti,
                                      print_stat = print_stat,
                                      multiClass = multiClass)
        dict_metrics["CGBRF"] = metrics
    
    return dict_metrics 

def dfToVisualization(df,classifiers,feat,feat_lab,
                      onehotplease= False,
                      averageMulti = "weighted",
                      print_stat = False, 
                      multiClass = False
                     ):
    dict_predictors = {}
    #input
    if not onehotplease: 
        X = df[feat]
    else:
        X = pd.get_dummies(df[feat])

    # target
    y = df[target]

    #split train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=2)

    dict_metrics = trainAndCompare(X_train, X_test, y_train, y_test,
                                   classifiers,
                                   #averageMulti = "weighted",
                                   print_stat = print_stat, 
                                   multiClass = multiClass)

    dict_predictors[feat_lab] = dict_metrics

    # set up a multiindex dictionary
    reformed_dict = {}
    for outerKey, innerDict in dict_predictors.items():
        for innerKey, values in innerDict.items():
            reformed_dict[(outerKey, innerKey)] = values

    multiIndex_df = pd.DataFrame(reformed_dict)
    results_df = multiIndex_df.T
    return results_df