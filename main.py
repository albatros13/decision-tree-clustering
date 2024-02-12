from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.preprocessing import Normalizer, MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from preprocess import *
from c45 import C45


def get_leaf_stats(clf):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    #node_depth = np.zeros(shape=n_nodes, dtype=np.int64)

    parents = np.full(n_nodes, -1)
    stack = [(0, -1)]
    leaves = []

    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        #node_depth[node_id] = parent_depth + 1
        if children_left[node_id] != children_right[node_id]:
            parents[children_left[node_id]] = node_id
            parents[children_right[node_id]] = node_id
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            leaves.append(node_id)
    return (leaves, parents)


def remove_best(clf, X, y):
    leaves, parents = get_leaf_stats(clf)

    impurity = clf.tree_.impurity
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    children_left = clf.tree_.children_left

    best_leave = -1
    min_impurity = 1
    # Choose the best (with minimal impurity)
    for leave in leaves:
        if impurity[leave] < min_impurity:
            min_impurity = impurity[leave]
            best_leave = leave

    expr = np.zeros(shape=X.shape[0], dtype=bool)
    if best_leave > 0:
        curr = best_leave
        while parents[curr] != -1:
            is_left = True if children_left[parents[curr]] == curr else False
            curr = parents[curr]
            value = threshold[curr]
            column_name = X.columns.values[feature[curr]]
            print(curr, impurity[curr], column_name, value)
            expr = expr | (X[column_name] >= value if is_left else X[column_name] < value)
    X = X.loc[expr]
    y = y.loc[expr]
    print(X.shape)
    return X, y


def tree_classifier(X, y, image_file_name="cls_tree.dot", impurity_decrease = 0.01):
    class_names = ["Negative", "Positive"]

    # X = select_features(X, y, 3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # clf = DecisionTreeClassifier(max_depth=3)
    clf = DecisionTreeClassifier(min_impurity_decrease=impurity_decrease, max_depth=4)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print_eval(y_test, y_pred)

    # Tree classifier for the whole dataset
    tree_clf = clf.fit(X, y)

    # Output linearized insight
    # print_insights(tree_clf)
    # r = export_text(tree_clf, feature_names=X.columns.values.tolist(), show_weights=True)
    # print(r)

    export_graphviz(clf, out_file=image_path(image_file_name),
        feature_names=X.columns.values, class_names=class_names, rounded=True, filled=True)

    print_feature_importances(X.columns.values, tree_clf.feature_importances_)
    return clf


def tree_classifier_filtered(X_n, X_p, target_name, skip_names=None):
    if skip_names is None:
        skip_names = []
    X_n, y_n = clean(X_n, target_name, skip_names)
    headers = X_n.columns.values
    # X_n = Normalizer().fit_transform(X_n)
    X_n = MinMaxScaler().fit_transform(X_n)
    X_p, y_p = clean(X_p, target_name, skip_names)
    # X_p = Normalizer().fit_transform(X_p)
    X_p = MinMaxScaler().fit_transform(X_p)
    y_p = np.ones(y_p.shape)

    clf = LocalOutlierFactor(n_neighbors=5, novelty=True)
    clf.fit(X_p)
    res_n = clf.predict(X_n)
    X_n_out = X_n[res_n == -1]
    y_n_out = y_n[res_n == -1]
    print(X_n_out.shape[0])

    # X_out = pd.DataFrame(np.vstack((X_n_out, X_p)), columns=headers)
    # y_out = np.vstack((y_n_out, y_p))
    #
    # # Distinct negatives vs distinct positives
    # clf.fit(X_n)
    # res_p = clf.predict(X_p)
    # X_p_out = X_p[res_p == -1]
    # y_p_out = y_p[res_p == -1]
    # X_p_in = X_p[res_p == 1]
    # y_p_in = y_p[res_p == 1]
    # print(X_p_out.shape[0])
    # X_out = pd.DataFrame(np.vstack((X_n_out, X_p_out)), columns=headers)
    # y_out = np.vstack((y_n_out, y_p_out))
    #
    # X_in = pd.DataFrame(np.vstack((X_n_in, X_p_in)), columns=headers)
    # y_in = np.vstack((y_n_in, y_p_in))
    #
    # tree_clf = tree_classifier(X_out, y_out, "tree-lof.dot", 0.005)
    #
    # # Test classifier on "similar" subsets
    # y_pred = tree_clf.predict(X_n)
    # print_eval(y_n, y_pred)


def tree_classifier_clustered(X_n, X_p, target_name, skip_names=None):
    if skip_names is None:
        skip_names = []
    X_n, y_n = clean(X_n, target_name, skip_names)
    headers = X_n.columns.values
    X_n = Normalizer().fit_transform(X_n)
    # X_n = MinMaxScaler().fit_transform(X_n)
    X_p, y_p = clean(X_p, target_name, skip_names)
    X_p = Normalizer().fit_transform(X_p)
    # X_p = MinMaxScaler().fit_transform(X_p)
    y_p = np.ones(y_p.shape)

    n = 2
    cls = KMeans(n_clusters=n)
    cls.fit_predict(X_n)
    X_n_cls = np.empty(n, dtype=object)
    y_n_cls = np.empty(n, dtype=object)
    clf = np.empty(n, dtype=object)
    for i in range(n):
        X_n_cls[i] = X_n[cls.labels_== i]
        y_n_cls[i] = y_n[cls.labels_== i]
        print(y_n_cls[i].shape)
        X_out = pd.DataFrame(np.vstack((X_n_cls[i], X_p)), columns=headers)
        y_out = np.vstack((y_n_cls[i], y_p))
        clf[i] = tree_classifier(X_out, y_out, "{}{}{}".format("tree-", i + 1, ".dot"), 0.005)


def test_titanic_filtered():
    df = load("titanic.csv")
    target_name = "Survived"
    X_n = select_rows(df, target_name, 0)
    X_p = select_rows(df, target_name, 1)
    tree_classifier_filtered(X_n, X_p, target_name)


def test_titanic_clustered():
    df = load("titanic.csv")
    target_name = "Survived"
    X_n = select_rows(df, target_name, 0)
    X_p = select_rows(df, target_name, 1)
    tree_classifier_clustered(X_n, X_p, target_name, [])


def test_mobile_clustered():
    df = load("vfmedium.csv")
    target_name = "response"
    X_n = select_rows(df, target_name, "Negative")
    X_p = select_rows(df, target_name, "Positive")
    tree_classifier_clustered(X_n, X_p, target_name,  ["id", "proposition", "cus_email", "cus_country"])


def test_mobile_filtered():
    df = load("vfmedium.csv")
    target_name = "response"
    X_n = select_rows(df, target_name, "Negative")
    X_p = select_rows(df, target_name, "Positive")
    tree_classifier_filtered(X_n, X_p, target_name,  ["id", "proposition", "cus_email", "cus_country"])


def test_titanic():
    X, y = load_and_clean("titanic.csv", "Survived", [], True)
    clf = tree_classifier(X, y, "titanic_tree.dot", 0.001)
    X, y = remove_best(clf, X, y)
    tree_classifier(X, y, "titanic_tree2.dot", 0.001)


def test_mobile():
    X, y = load_and_clean("vfmedium.csv", "response", ["id", "proposition", "cus_email", "cus_country"], True)
    clf = tree_classifier(X, y, "mobile_tree.dot", 0.001)
    X, y = remove_best(clf, X, y)
    tree_classifier(X, y, "mobile_tree2.dot", 0.001)


def test_mobile_dense_negatives():
    df = load("vfmedium.csv")

    X_n = select_rows(df, "response", "Negative")
    X_n, y_n = clean(X_n, "response", ["id", "proposition"]) #, "cus_email", "cus_country"])
    headers = X_n.columns.values
    X_n = Normalizer().fit_transform(X_n)

    clf = LocalOutlierFactor(n_neighbors=3)
    res_n = clf.fit_predict(X_n)
    X_n_out = X_n[res_n == -1]
    y_n_out = y_n[res_n == -1]
    X_n_in = X_n[res_n == 1]
    y_n_in = y_n[res_n == 1]

    y_n_out = np.zeros(y_n_out.shape)
    y_n_in = np.ones(y_n_in.shape)

    print(X_n_out.shape)
    print(X_n_in.shape)

    X = pd.DataFrame(np.vstack((X_n_out, X_n_in)), columns=headers)
    y = np.vstack((y_n_out, y_n_in))
    tree_classifier(X, y)


# These are tests to work with German bank credit data (reformatted and stored in csv format)
# https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

def test_german():
    X, y = load_and_clean("german-credit.csv", "CHURN")
    clf = tree_classifier(X, y, "german_tree.dot")
    X, y = remove_best(clf, X, y)
    tree_classifier(X, y, "german_tree2.dot")


def test_german_filtered():
    df = load("german-credit.csv")
    target_name = "CHURN"
    X_n = select_rows(df, target_name, "Negative")
    X_p = select_rows(df, target_name, "Positive")
    print(X_n.shape)
    print(X_p.shape)
    tree_classifier_filtered(X_n, X_p, target_name)


def test_german_numeric_filtered():
    df = load("german-numeric.csv")
    target_name = "CHURN"
    X_n = select_rows(df, target_name, 1)
    X_p = select_rows(df, target_name, 2)
    print(X_n.shape)
    print(X_p.shape)
    tree_classifier_filtered(X_n, X_p, target_name)


def test_mobile_c45():
    # c1 = C45(resource_path("iris.data"), resource_path("iris.names"))
    gen_data_for_c45("vfmedium.csv", "mobile", "response", ["Positive", "Negative"], ["id", "proposition"])
    c1 = C45(resource_path("mobile.data"), resource_path("mobile.names"))
    c1.fetch_data()
    # TODO fix me - the algorithm does not work with string values
    c1.preprocess_data()
    c1.generate_tree()
    c1.print_tree()


# Run a selected test

# test_mobile()

test_titanic()
