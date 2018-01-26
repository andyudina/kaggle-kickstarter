"""
Create a model to predict if project will achieve it goals
"""
import logging

import numpy as np
import pandas as pd
import plotly as py
from plotly.graph_objs import Bar, Scatter
from scipy import sparse


logging.basicConfig(level=logging.DEBUG)


ks_projects_2018_df = pd.read_csv('ks-projects-201801_cleaned_for_classification.csv')
text_features_names = sparse.load_npz(
    'ks-projects-201801_cleaned_for_classification-headers.npz')
text_features = sparse.load_npz(
    'ks-projects-201801_cleaned_for_classification.npz')

def _is_nan(val):
    import math
    return type(val) == float and math.isnan(val)

# Tag successfull projects 
def is_project_successfull(project):
    """
    Check if project is successfull. For unknown status projects returns None
    """
    SUCCESSFULL_STATUS = 'successful'
    UNSUCCESSFULL_STATUSES = ['failed', 'canceled']
    if project.state == SUCCESSFULL_STATUS:
        return True
    if project.state in UNSUCCESSFULL_STATUSES:
        return False
    return None


# Add time requested for money collection in days
def _parse_datetime(dt_str):
    from dateutil.parser import parse
    return parse(dt_str)

def get_time_requested_for_money_collection(project):
    """
    Calculate number of days requested for money collection
    """
    return (
        _parse_datetime(project.deadline) - _parse_datetime(project.launched)  
    ).days


# Convert goal sum to usd
def _convert_amount(
        amount, 
        initial_currency, target_currency,
        conversion_rate_date):
    # This queries 3d party server and is slow
    from currency_converter import CurrencyConverter
    converter = CurrencyConverter(
        # use best effors strategy for calculating exact conversion rate
        fallback_on_wrong_date=True,
        fallback_on_missing_rate=True)
    return converter.convert(
        amount, 
        initial_currency, 
        target_currency,
        date=conversion_rate_date)

def _convert_amount_using_fixed_rate(
        amount, 
        initial_currency,
        *args):
    """
    Convert currencies using hardcoded rate 
    # (took one for 25.01.2017 - better to take at least median for project years)
    Used to save time & in cases when 3d party server is not available
    """
    CURRENCIES_RATES_TO_USD = {
        'GBP': 1.43,
        'EUR': 1.25,
        'CAD': 0.81,
        'AUD': 0.81,
        'NOK': 0.13,
        'MXN': 0.054,
        'SEK': 0.13,
        'NZD': 0.74,
        'CHF': 1.07,
        'DKK': 0.17,
        'HKD': 0.13,
        'SGD': 0.77,
        'JPY': 0.0092,
    }
    # Usually decimals work better for money. 
    # Here accuarcy is not needed though
    return CURRENCIES_RATES_TO_USD[initial_currency] * amount

def convert_goal_sum_to_usd(project):
    """
    Convert goal sum in foreign currency to USD
    """
    USD_CURRENCY = 'USD'
    if project.currency == USD_CURRENCY:
        return project.goal
    return _convert_amount_using_fixed_rate(
        project.goal,
        project.currency)


# Add pledged rate
def calculate_pledged_rate(project):
    """
    Calculate the percentage of goal amount which project was able to collect
    """
    return float(project['usd pledged'] or 0) / project.goal_usd


# Add month, date of week of launch
def get_launch_date_of_week(project):
    """
    Get date of week from launch date
    """
    return _parse_datetime(project.launched).weekday()

def get_launch_month(project):
    """
    Get month from launch date
    """
    return _parse_datetime(project.launched).month


def preprocess_initial_dataset(ks_projects_2018_df):
    """
    Add new features & filter out non informative values.
    Save to new csv file
    """
    logging.info('Tag successfull and unsuccessfull projects')
    ks_projects_2018_df.is_successfull = ks_projects_2018_df.apply(
        is_project_successfull,
        axis=1)

    # Filter out projects with unknown status
    logging.info('Filter out projects with unknown status')
    ks_projects_2018_df = ks_projects_2018_df[
        ks_projects_2018_df.is_successfull.notnull()
    ]

    logging.info('Add time requested for money collection in days')
    ks_projects_2018_df.days_to_collect_money = ks_projects_2018_df.apply(
        get_time_requested_for_money_collection,
        axis=1)

    logging.info('Convert goal sum to usd')
    ks_projects_2018_df.goal_usd = ks_projects_2018_df.apply(
        convert_goal_sum_to_usd,
        axis=1)

    logging.info('Add pledged rate')
    ks_projects_2018_df.pledged_rate = ks_projects_2018_df.apply(
        calculate_pledged_rate,
        axis=1)

    logging.info('Add month, date of week of launch')
    ks_projects_2018_df.day_of_week_launched = ks_projects_2018_df.apply(
        get_launch_date_of_week,
        axis=1)

    ks_projects_2018_df.month_launched = ks_projects_2018_df.apply(
        get_launch_month,
        axis=1)

    # Save intermediate results
    logging.info('Save intermediate results')
    ks_projects_2018_df.to_csv(
        'ks-projects-201801_cleaned.csv',
        index=False)

def preprocess_initial_dataset_(ks_projects_2018_df):
    """
    Add new features & filter out non informative values.
    Save to new csv file
    """
    def _generate_new_features_for_project(project):
        """
        Helper to generate all needed features for project
        """
        project['is_successfull'] = is_project_successfull(project)
        project['days_to_collect_money'] = \
            get_time_requested_for_money_collection(project)
        project['goal_usd'] = convert_goal_sum_to_usd(project)
        project['day_of_week_launched'] = get_launch_date_of_week(project)
        project['month_launched'] = get_launch_month(project)
        return project


    logging.info('Add new features to dataframe')
    ks_projects_2018_df =  ks_projects_2018_df.apply(
        _generate_new_features_for_project,
        axis=1)

    # Filter out projects with unknown status
    logging.info('Filter out projects with unknown status')
    ks_projects_2018_df = ks_projects_2018_df[
        ks_projects_2018_df.is_successfull.notnull()
    ]

    # Save intermediate results
    logging.info('Save intermediate results')
    ks_projects_2018_df.to_csv(
        'ks-projects-201801_cleaned.csv',
        index=False)


def clean_days_to_collect_money(ks_projects_2018_df):
    """
    Filter out unrealistically long projects and save to cleaned csv file
    """
    MAXIMUM_SANE_PERIOD_TO_COLLECT_MONEY = 92
    ks_projects_2018_df = ks_projects_2018_df[
        ks_projects_2018_df.days_to_collect_money <=\
        MAXIMUM_SANE_PERIOD_TO_COLLECT_MONEY]
    ks_projects_2018_df.to_csv(
        'ks-projects-201801_cleaned.csv',
        index=False)


# Visualise distributions
def plot_success_descrete_distribution(
        df, success_metric, group_by_metric):
    """
    Plot success and fails rate for different values
    of "group_by" metric
    """
    success_distribution = df.groupby(
        [group_by_metric],
        sort=True)[success_metric].mean()
    success_distribution = success_distribution.to_frame()
    x = success_distribution.index
    y = success_distribution[success_metric]
    data = Bar(x = x, y = y)
    py.offline.plot(
        [data],
        filename='%s-success_distribution' % group_by_metric)

def plot_success_numeric_distribution(
        df, success_metric, target_metric):
    """
    Plot success and fails for traget metric
    """
    y = df[target_metric]
    x = df[success_metric].astype(int)
    data = Scatter(x = x, y = y, mode = 'markers')
    py.offline.plot(
        [data],
        filename='%s-success_distribution' % target_metric)

CATEGORICAL_FEATURES = [
    'category',
    'main_category',
    'country',
]

NUMERICAL_DESCRETE_FEATURES = [
    'day_of_week_launched',
    'month_launched'
]

NUMERICAL_FEATURES = [
    'days_to_collect_money',
    'goal_usd',
    'backers',
]

# Features that seams to have not unifified distribution
# found on visualisation
SIGNIFICANT_NUMERIC_FEATURES = [
    'month_launched',
    'goal_usd'
]

TARGET_COLUMN = 'is_successfull'


def plot_success_distributions(ks_projects_2018_df):
    """
    Plot success distributions for numerical and descrete features
    """
    # From visualisation:
    # There is small correlation between month of launch, categories and goal sum
    # and success results. Number of backers also correlates with positive result
    # but I'd use it as a target metric 
    # (to get more money one need to get more people involved - isn't it the whole platform mechanic)
    for metric in CATEGORICAL_FEATURES + NUMERICAL_DESCRETE_FEATURES:
        plot_success_descrete_distribution(
            ks_projects_2018_df,
           'is_successfull',
            metric)


    for metric in NUMERICAL_FEATURES:
        plot_success_numeric_distribution(
            ks_projects_2018_df,
            'is_successfull',
            metric)


# Extract text features from name
def _tokenise(name):
    from nltk.tokenize import word_tokenize
    try:
        return word_tokenize(name)
    except UnicodeDecodeError: 
        return word_tokenize(name.decode('utf-8'))

def _filter_out_stop_words(tokens):
    from nltk.corpus import stopwords
    import string
    tokens = [token.lower() for token in tokens]
    stop_list = stopwords.words('english') + list(string.punctuation)
    return filter(
            lambda token: token not in stop_list,
            tokens)

def _normalise(tokens):
    def _stem(w):
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
        try:
            return stemmer.stem(token)
        except UnicodeDecodeError:
            return stemmer.stem(token.decode('utf-8'))

    def _is_token_significant(token):
        MIN_SIGNIFICANT_TOKEN_LENGTH = 2
        TOKEN_STOP_WORDS = ['\'s', '\'re']
        return len(token) >= MIN_SIGNIFICANT_TOKEN_LENGTH and \
            token not in TOKEN_STOP_WORDS

    stemmed_tokens = [
        _stem(token)
        for token in tokens]
    return [
        token for token in stemmed_tokens
        if _is_token_significant(token)
    ]


def normalise_project_name(project):
    """
    Extract stems from project name
    """
    # TODO: add n-gramms if needed
    name = project['name']
    if not name or _is_nan(name): return []
    tokens = _tokenise(name)
    tokens_filtered = _filter_out_stop_words(tokens)
    return _normalise(tokens_filtered)

def normalise_project_names(ks_projects_2018_df):
    """
    Add normalised project name column to dataframe and save it to csv
    """ 
    ks_projects_2018_df['normalised_name'] = ks_projects_2018_df.apply(
        normalise_project_name,
        axis=1)
    ks_projects_2018_df.to_csv(
        'ks-projects-201801_cleaned.csv',
        index=False)


# Caluclate tf-idf value of each word
def calculate_tf_idf_for_project_names(names_sequence):
    """
    Calculate tf_idf metric for each stem for each name.
    Returns dataframe that can be matched by index
    """
    def _generate_columns_from_vocabulary(vocabulary):
        return [
            v[0] for v in
            sorted(
                vocabulary.items(),
                key=lambda v: v[-1])]

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tf_idf_metrics = vectorizer.fit_transform(names_sequence.as_matrix())
    columns = _generate_columns_from_vocabulary(vectorizer.vocabulary_)
    return tf_idf_metrics, columns


# Binarise categorical features
def binarise_categorical_features(
        df, categorical_features):
    """
    Binarise categocrical features of dataset
    Save new dataset to csv
    """
    from sklearn.feature_extraction import DictVectorizer
    v = DictVectorizer(sparse=False)
    D = [
        val for index, val in 
        sorted(
            df[categorical_features].T.to_dict().items(),
            key=lambda x: x [0])]
    X = v.fit_transform(D)
    binarised_df = pd.DataFrame(
        data=X, 
        columns=v.feature_names_)
    df = pd.concat(
        (df, binarised_df), 
        axis=1,  join='inner')
    return df, v.feature_names_


def normalise_numeric_features_inplace(
        df, features):
    """
    Use "soft normalisation" for each numeric feature
    (amount - mean)  / 2 std. Don't handle cases with std == 0
    """
    def _normalise_numeric_feature(df, feature):
        mean = df[feature].mean()
        std = df[feature].std()
        df[feature] = df.apply(
            lambda project: abs(project[feature] - mean) / (2 * std),
            axis=1)
        if df[feature].max() >= 1 or df[feature].min() <= 0:
            # rollback to min-max normalisation
            df[feature] = \
                (df[feature] - df[feature].min()) / (df[feature].max() - df[feature].min())

    for feature in features:
        _normalise_numeric_feature(df, feature)


def merge_text_features_and_original_dataset(
        tf_idf_metrics, text_features,
        initial_df, columns_to_concat):
    """
    Merge to text features matrix and original dataset
    using sparse matrix
    """
    features = sparse.hstack(
        (
            tf_idf_metrics,
            sparse.csr_matrix(
                initial_df[columns_to_concat].as_matrix())
        )).todense()

    columns = np.hstack(
        (
            text_features,
            columns_to_concat
        ))
 
    return features, columns


def prepare_initial_dataset_for_classification_and_save_to_csv(df):
    """
    Binarised features to dataset. 
    Normalise numeric features
    Remove not needed features
    """
    logging.info('Binarise categorical features')
    df, binarised_features = binarise_categorical_features(
        df, CATEGORICAL_FEATURES)

    logging.info('Normalise numeric features')
    normalise_numeric_features_inplace(
        df, SIGNIFICANT_NUMERIC_FEATURES)

    logging.info('Save to csv without text')
    COLUMN_NEEDED_FOR_TEXT_ANALYSIS = 'normalised_name'
    df.to_csv(
        'ks-projects-201801_cleaned_for_classification.csv',
        columns=binarised_features + \
            SIGNIFICANT_NUMERIC_FEATURES + \
            [TARGET_COLUMN, COLUMN_NEEDED_FOR_TEXT_ANALYSIS],
        index=False)
    return df

def _save_columns_order_to_csv(final_columns):
    """
    Helper to create npz file with text columns names
    """
    sparse_matrix = sparse.coo_matrix(final_columns)
    sparse.save_npz(
        'ks-projects-201801_cleaned_for_classification-headers.npz',
        sparse_matrix)

def _store_text_features_as_sparce_matrix(matrix):
    """
    Save text features as matrix
    """
    sparse.save_npz(
        'ks-projects-201801_cleaned_for_classification.npz', 
        matrix)

def generate_text_features_and_save_to_csv(df):
    """
    Add text features to dataset and save to csv
    """
    logging.info('Calculate tf idf features')
    tf_idf_metrics, text_features = \
        calculate_tf_idf_for_project_names(df.normalised_name)


    logging.info('Save text data as sparce matrix')
    _save_columns_order_to_csv(text_features)
    _store_text_features_as_sparce_matrix(tf_idf_metrics)


def get_dataset_for_classification(
        tf_idf_metrics, text_features, df):
    """
    Merge text features sparce array and other features dataset.
    Return sparse matrix with features, target column and column names
    """
    # get list from coo_matrix of text features
    text_features = list(text_features.data)

    # generate binarised features names
    binarised_features = [
        column for column in df.columns.tolist()
        if any(
            feature in column 
            for feature in CATEGORICAL_FEATURES)]

    # merge text features with original dataset effeciently
    # conversion to int is needed for successful merge using sparse matrix
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    logging.info('Get binarised columns names')
    binarised_features = [
        column for column in df.columns.tolist()
        if any(
            feature in column 
            for feature in CATEGORICAL_FEATURES)]
  
    target_column = df[TARGET_COLUMN].as_matrix()

    logging.info('Merge features and columns')
    features, columns = merge_text_features_and_original_dataset(
        tf_idf_metrics, text_features,
        df, binarised_features + \
        SIGNIFICANT_NUMERIC_FEATURES)

    return features, columns, target_column


features, column_names, target_column = \
    get_dataset_for_classification(
        text_features, text_features_names, ks_projects_2018_df)


# Teach SVM calssifier
def teach_svm_classifier(X, y):
    """
    Teach SVC classifier and validate via cross validation
    Return accuracy and classifier
    """
    import itertools
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score

    def _teach_and_validate_svm_classifier(
            C, gamma, X, y):
        clf = SVC(C=C, gamma=gamma)
        NUMBER_OF_CROSS_VAL_TRIES = 3
        scores = cross_val_score(clf, X, y, cv=NUMBER_OF_CROSS_VAL_TRIES)
        return \
            clf.fit(X, y), \
            np.mean(scores)
            
    logging.info('Teach classifier')
    C_s = [
        0.1, 1, 10, 100, 1000]
    gammas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    c_and_gammas = itertools.product(C_s, gammas)

    # run classification with different SVM params
    classification_results = []
    for C, gamma in c_and_gammas:
        logging.info('Test C and gamma:%d / %d' % (C, gamma))
        classifier, accuracy = _teach_and_validate_svm_classifier(
            C, gamma, X, y)
        classification_results.append(
            {
                'clf': classifier,
                'accuracy': accuracy,
                'C': C,
                'gamma': gamma
            })

    classification_results = sorted(
           classification_results,
           key=lambda res: res['accuracy'],
           reversed=True)

    # print top results
    for res in classification_results[:5]:
        logging.info('C', res['C'])
        logging.info('gamma', res['gamma'])
        logging.info('accuracy', res['accuracy'])
    # return best classifier
    return classification_results[0]['clf']

teach_svm_classifier(
    features,
    target_column)