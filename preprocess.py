from utils import *
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest


def gen_data_for_c45(file_name, output_file, target_name, class_names, skip_names=None):
    df = pd.read_csv(resource_path(file_name), sep=',')
    y = df[target_name]

    # FEATURES
    X = df.drop([target_name] + skip_names, axis=1)
    X = X.dropna(axis=1, how='all')
    X.join(y).to_csv(resource_path(output_file + ".data"), sep=',')

    dt_names = [', '.join(class_names)]
    # The format - column_name : column_type which is either "continuous" or a comma-separated list of discrete values
    for name, values in X.iteritems():
        type = ', '.join(map(str, pd.unique(values))) if X.dtypes[name] == "object" else "continuous"
        dt_names.append(name + " : " + type)
    with open(resource_path(output_file + ".names"), 'w', newline='\n') as myfile:
        for line in dt_names:
            myfile.write('%s\n' % line)


def load(file_name):
    model_file = resource_path(file_name)
    if not os.path.isfile(model_file):
        print("Model file not found!")
        return None, None

    return pd.read_csv(model_file, sep=',')


def clean(df, target_name, skip_names=None, useVariance=False):
    if skip_names is None:
        skip_names = []
    if not target_name:
        print("Target column not specified")
        return None, None

    # Pre-processing pipelines
    categorical_pipeline = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value="Missing", copy=False)),
        ('encoder', OrdinalEncoder())
        # ('encoder', OneHotEncoder())
    ])
    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='mean', copy=False))
    ])

    # TARGET
    label_enc = LabelEncoder()
    y = label_enc.fit_transform(df[target_name])
    y = pd.DataFrame(y, columns=[target_name])

    # FEATURES
    X = df.drop([target_name] + skip_names, axis=1)
    X = X.dropna(axis=1, how='all')

    # print(X.dtypes)

    # Categorical pre-processing
    X_categorical = X.select_dtypes(include=['object']).copy()
    if len(X_categorical.columns) > 0:
        X_categorical_encoded = categorical_pipeline.fit_transform(X_categorical)
        X_categorical = pd.DataFrame(X_categorical_encoded, columns=X_categorical.columns.values)

    # TODO return Category encoding
    # categorical_pipeline["encoder"].categories_

    # Numeric pre-processing
    X_numeric = X.select_dtypes(include=['float64', 'int64']).copy()
    X_numeric_filtered = numeric_pipeline.fit_transform(X_numeric)
    if useVariance:
        filter = VarianceThreshold(0.2)
        X_numeric_filtered = filter.fit_transform(X_numeric_filtered)
        X_numeric = pd.DataFrame(X_numeric_filtered, columns=X_numeric.columns[filter.get_support(indices=True)])
    else:
        X_numeric = pd.DataFrame(X_numeric_filtered, columns=X_numeric.columns.values)

    X = pd.concat([X_numeric, X_categorical], axis=1, sort=False)
    return X, y


def load_and_clean(file_name, target_name, skip_names=None, useVariance=False):
    if skip_names is None:
        skip_names = []
    df = load(file_name)
    return clean(df, target_name, skip_names, useVariance)


def select_features(X, y, k):
    selector = SelectKBest(k=k)
    X_selected = selector.fit_transform(X,y)
    X_selected = pd.DataFrame(X_selected, columns=X.columns[selector.get_support(indices=True)])
    return X_selected


def select_rows(df, target_name, target_value):
    df_filtered = pd.DataFrame(df.loc[df[target_name] == target_value], columns=df.columns.values)
    return df_filtered

# The decision estimator has an attribute called tree_  which stores the entire
    # tree structure and allows access to low level attributes. The binary tree
    # tree_ is represented as a number of parallel arrays. The i-th element of each
    # array holds information about the node `i`. Node 0 is the tree's root.
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node
def print_insights(estimator):
    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    print("The binary tree structure has %s nodes:"
          % n_nodes)
    for i in range(n_nodes):
        if is_leaves[i]:
            print("%snode=%s leaf node." % (node_depth[i] * "\t", i))
        else:
            print("%snode=%s test node: go to node %s if X[:, %s] <= %s else to "
                  "node %s."
                  % (node_depth[i] * "\t",
                     i,
                     children_left[i],
                     feature[i],
                     threshold[i],
                     children_right[i],
                     ))
    print()

