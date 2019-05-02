from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
X, y = load_iris(return_X_y=True)
for k in tqdm(range(100000)):
    clf = LogisticRegression(random_state=k, solver='lbfgs',
                             multi_class='multinomial').fit(X, y)
from joblib import dump, load

dump(clf, 'outputs/mymodel.joblib')