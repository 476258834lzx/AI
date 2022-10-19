from sklearn.metrics import r2_score,explained_variance_score

if __name__ == '__main__':
    y=[[5,3,2,7],[1,2,4,5]]
    pred=[[4.9,3,1,8],[1,2,5,3]]

    print(r2_score(y,pred))
    print(explained_variance_score(y,pred))