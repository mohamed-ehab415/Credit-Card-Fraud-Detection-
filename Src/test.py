from evalution import  *


if __name__=='__main__':
    path_test=r'C:\Users\lap shop\OneDrive\Documents\machin lear\project 2\split\test.csv'
    test=load_data(path_test)
    X_test,t_test=split_data(test)
    X_test['Time_log'] = np.log1p(X_test['Time'])
    X_test['Amount_log'] = np.log1p(X_test['Amount'])

    selected_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V11', 'V12',
                        'V14', 'V16', 'V17', 'V18', 'V19', 'V21', 'Time_log', 'Amount_log']
    X_test=X_test[selected_columns]

    ## i choose voting model is the best
    voting = joblib.load('voting_classifier_model.pkl')
    evaluate_model(voting,X_test,t_test,threshold=0.65,data='testing')


#testing F1 Score for VotingClassifier is : 0.8528
#testing PR AUC for VotingClassifier is : 0.8531
