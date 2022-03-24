## Import chdir function from os module
from os import chdir

## Change working directory
chdir(r'C:\Users\COLLABERA TECH\PycharmProjects\boston')

## Streamlit for creating forms
import streamlit as st

## Packages for data manipulation
import numpy as np
import pandas as pd

## Import sklearn
import sklearn

## sklearn classes for machine learning and column transformers
from sklearn.compose import make_column_selector
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

## Machine Learning models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor

## Create requirements.txt file
packages = [np, pd, sklearn, st]
with open('requirements.txt', 'w') as f:
    for package in packages:
        f.write(package.__name__ + '==' + package.__version__)
        f.write('\n')

## Load data and split features and response
boston = pd.read_csv('Boston.csv', index_col=0)
X = boston.drop(['black', 'medv'], axis=1)
X.tax = X.tax.astype(float)
y = boston.medv.values

## Creating data preprocessing pipeline
## Define column selector
cat_selector = make_column_selector(dtype_include='int64')
num_selector = make_column_selector(dtype_include='float64')

## Define data transformers
onehot_enc = OneHotEncoder(handle_unknown="ignore")
scaler = StandardScaler()
pca = PCA(3)

## Define the column transformer
transformer = make_column_transformer(
    (onehot_enc, cat_selector), (scaler, num_selector))

## Define steps in preprocessing
steps = [('transformer', transformer), ('pca', pca)]

## Define pipeline
preprocessing_pipe = Pipeline(steps=steps)

## Fit the data to the pipeline
X_preprocessed = preprocessing_pipe.fit_transform(X)

## Transform the output variable
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

## Machine learning models for predictive analytics
## Initialize models
linear_regression = LinearRegression()
lasso = Lasso(alpha=0.0788, max_iter=100, random_state=42)
ridge = Ridge(alpha=0.3162, max_iter=100, random_state=42, solver='saga')
svr = SVR(C=10.0, gamma='auto')
tree = DecisionTreeRegressor(criterion='absolute_error', max_depth=5, max_features='auto', random_state=42)
forest = RandomForestRegressor(max_depth=9, max_features='sqrt', random_state=42)
xgb = GradientBoostingRegressor(criterion='squared_error', loss='absolute_error', random_state=42)

## Create stacked regressor
## Combine models
estimators = [linear_regression, lasso, ridge, svr, tree, xgb, forest]
estimators = [(type(x).__name__, x) for x in estimators]

## Create the stacked regressor; use RandomForestRegressor as final_estimator
stacking_regressor = StackingRegressor(
    estimators=estimators[:-1], final_estimator=estimators[-1][1], cv=10)

## Model fitting on all models
for i, estimator in enumerate(estimators):
    estimators[i] = estimator[1].fit(X_preprocessed, y_scaled)

## Model fitting on stacked regressor
stacking_regressor.fit(X_preprocessed, y_scaled)

## Browser tab title
st.set_page_config(
    page_title="Boston House Price estimator",
    page_icon="📊", initial_sidebar_state="expanded"
)

## Function to get min, median, or max
def get_stat(series, stat, places=2):
    if stat == 'min':
        return_value = series.min()
    elif stat=='median':
        return_value = series.median()
    else:
        return_value = series.max()
    return round(float(return_value), places)


## Page title
st.write('# 📊 Boston House Price Estimator')

## Form
with st.form('boston_estimator_form'):

    st.write('Enter what you want your house to be, and click `Estimate` to get the estimated price.')

    ## User inputs
    crim = st.slider('Crime rate of the area:',
                     value=get_stat(X.crim, 'median'), format='%f', step=0.01,
                     min_value=get_stat(X.crim, 'min'), max_value=get_stat(X.crim, 'max'))
    zn = st.slider('Proportion of residential land zoned for lots over 25k sq.ft.:',
                   value=get_stat(X.zn, 'median'), format='%f', step=0.01,
                   min_value=get_stat(X.zn, 'min'), max_value=get_stat(X.zn, 'max'))
    indus = st.slider('Proportion of non-retail business acres per town:',
                      value=get_stat(X.indus, 'median'), format='%f', step=0.01,
                      min_value=get_stat(X.indus, 'min'), max_value=get_stat(X.indus, 'max'))
    nox = st.slider('Nitric oxides concentration (parts per 10 million):',
                    value=get_stat(X.nox, 'median'), format='%f', step=0.01,
                    min_value=get_stat(X.nox, 'min'), max_value=get_stat(X.nox, 'max'))
    rm = st.slider('Average number of rooms per dwelling:',
                   value=get_stat(X.rm, 'median'), format='%f', step=0.01,
                   min_value=get_stat(X.rm, 'min'), max_value=get_stat(X.rm, 'max'))
    age = st.slider('Proportion of owner-occupied units built prior to 1940',
                    value=get_stat(X.age, 'median'), format='%f', step=0.01,
                    min_value=get_stat(X.age, 'min'), max_value=get_stat(X.age, 'max'))
    dis = st.slider('Weighted distances to five Boston employment centres',
                    value=get_stat(X.dis, 'median'), format='%f', step=0.01,
                    min_value=get_stat(X.dis, 'min'), max_value=get_stat(X.dis, 'max'))
    tax = st.slider('Full-value property-tax rate per 10,000 USD',
                    value=get_stat(X.tax, 'median'), format='%f', step=0.01,
                    min_value=get_stat(X.tax, 'min'), max_value=get_stat(X.tax, 'max'))
    ptratio = st.slider('Pupil-teacher ratio by town',
                        value=get_stat(X.ptratio, 'median'), format='%f', step=0.01,
                        min_value=get_stat(X.ptratio, 'min'), max_value=get_stat(X.ptratio, 'max'))
    lstat = st.slider('% lower status of the population',
                        value=get_stat(X.lstat, 'median'), format='%f', step=0.01,
                        min_value=get_stat(X.lstat, 'min'), max_value=get_stat(X.lstat, 'max'))
    rad = st.selectbox('Accessibility index to radial highways (higher means closer to highways):',
                       options=sorted(X.rad.unique()))

    chas = st.checkbox('Close to Charles River?')

    submitted = st.form_submit_button('Estimate')

if submitted:
    ## Create dataframe
    new_data = pd.DataFrame([{'crim':crim, 'zn':zn, 'indus':indus, 'chas':chas*1,
                              'nox':nox, 'rm':rm, 'age':age, 'dis':dis, 'rad':rad,
                              'tax':tax, 'ptratio':ptratio, 'lstat':lstat}])

    ## Transform the data using pipeline
    new_data_transformed = preprocessing_pipe.transform(new_data)

    ## Print the entered data as dataframe
    st.write('## New data entered')
    st.dataframe(new_data)

    ## Print predictions from different models
    st.write('## Predictions')
    predictions = []
    for estimator in estimators:
        prediction = estimator.predict(new_data_transformed)
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
        prediction = float(prediction[0])
        predictions.append(prediction)
        st.write(type(estimator).__name__ + ': {:.4f}'.
                 format(prediction))

    prediction = stacking_regressor.predict(new_data_transformed)
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1))
    prediction = float(prediction[0])
    predictions.append(prediction)
    st.write('StackedRegressor: {:.4f}'.format(prediction))

    st.write('**Average Prediction: {:.4f} $\pm$ {:.4f}**'.format(
        float(np.mean(predictions)), float(np.std(predictions))
    ))

    st.write('**Median Prediction: {:.4f}**'.format(
        float(np.median(predictions))
    ))