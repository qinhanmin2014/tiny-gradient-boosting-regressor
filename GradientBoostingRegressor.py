import numpy as np


# Reference: XGBoost: A Scalable Tree Boosting System
# (Algorithm 1 Exact Greedy Algorithm for Split Finding)
class TreeNode:
    def __init__(self):
        self.is_leaf = False
        self.split_feature = None
        self.split_val = None
        self.left_child = None
        self.right_child = None
        self.weight = None


class Tree:
    def __init__(self):
        self.root = None

    def _calc_weight(self, grad, hess, reg_lambda):
        return -np.sum(grad) / (np.sum(hess) + reg_lambda)  # equation (5)

    def _calc_loss_reduction(self, G, H, G_L, H_L, G_R, H_R, reg_lambda):
        return (np.square(G_L) / (H_L + reg_lambda) +
                np.square(G_R) / (H_R + reg_lambda) -
                np.square(G) / (H + reg_lambda))  # equation (7)

    def fit(self, X, y, grad, hess, shrinkage_rate, max_depth, reg_lambda,
            min_loss_reduction):
        self.root = TreeNode()
        self._fit(self.root, X, y, grad, hess, shrinkage_rate, max_depth,
                  reg_lambda, min_loss_reduction, 1)
        return self

    def _fit(self, node, X, y, grad, hess, shrinkage_rate, max_depth,
             reg_lambda, min_loss_reduction, depth):
        if depth >= max_depth:
            node.is_leaf = True
            node.weight = self._calc_weight(grad, hess,
                                            reg_lambda) * shrinkage_rate
            return
        G, H = np.sum(grad), np.sum(hess)
        best_loss_redunction = -np.inf
        best_split_feature, best_split_val = None, None
        best_left_ids, best_right_ids = None, None
        for i in range(X.shape[1]):
            G_L, H_L = 0, 0
            sorted_ids = np.argsort(X[:, i])
            for j in range(X.shape[0]):
                G_L += grad[sorted_ids[j]]
                H_L += hess[sorted_ids[j]]
                G_R = G - G_L
                H_R = H - H_L
                cur_loss_reduction = self._calc_loss_reduction(
                    G, H, G_L, H_L, G_R, H_R, reg_lambda)
                if cur_loss_reduction > best_loss_redunction:
                    best_loss_redunction = cur_loss_reduction
                    best_split_feature = i
                    best_split_val = X[sorted_ids[j], i]
                    best_left_ids = sorted_ids[:j + 1]
                    best_right_ids = sorted_ids[j + 1:]
        if best_loss_redunction < min_loss_reduction:
            node.is_leaf = True
            node.weight = self._calc_weight(grad, hess,
                                            reg_lambda) * shrinkage_rate
            return
        node.split_feature = best_split_feature
        node.split_val = best_split_val
        node.left_child = TreeNode()
        self._fit(node.left_child, X[best_left_ids], y[best_left_ids],
                  grad[best_left_ids], hess[best_left_ids], shrinkage_rate,
                  max_depth, reg_lambda, min_loss_reduction, depth + 1)
        node.right_child = TreeNode()
        self._fit(node.right_child, X[best_right_ids], y[best_right_ids],
                  grad[best_right_ids], hess[best_right_ids], shrinkage_rate,
                  max_depth, reg_lambda, min_loss_reduction, depth + 1)

    def predict(self, x):
        return self._predict(self.root, x)

    def _predict(self, node, x):
        if node.is_leaf:
            return node.weight
        else:
            if x[node.split_feature] <= node.split_val:
                return self._predict(node.left_child, x)
            else:
                return self._predict(node.right_child, x)


class GradientBoostingRegressor:
    def __init__(self, max_depth=5, learning_rate=0.8,
                 n_estimators=20, reg_lambda=1, min_loss_redunction=0.1):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.reg_lambda = reg_lambda
        self.min_loss_redunction = min_loss_redunction

    def _calc_loss(self, X, y, trees):
        y_pred = self._predict(X, trees)
        return np.mean(np.square(y_pred - y))  # L2 loss

    def _calc_grad_hess(self, X, y, trees):
        if len(trees) == 0:
            grad = -2 * y
        else:
            y_pred = self._predict(X, trees)
            grad = 2 * (y_pred - y)
        hess = np.full(X.shape[0], 2)
        return grad, hess

    def fit(self, X, y, X_val, y_val, early_stopping_rounds=5):
        trees = []
        shrinkage_rate = 1
        best_iteration = None
        best_val_loss = np.inf
        for iter_cnt in range(1, self.n_estimators + 1):
            grad, hess = self._calc_grad_hess(X, y, trees)
            cur_tree = Tree().fit(X, y, grad, hess, shrinkage_rate,
                                  self.max_depth, self.reg_lambda,
                                  self.min_loss_redunction)
            trees.append(cur_tree)
            shrinkage_rate *= self.learning_rate
            train_loss = self._calc_loss(X, y, trees)
            val_loss = self._calc_loss(X_val, y_val, trees)
            print("Iteration: {}  Train Loss: {:.8f}  "
                  "Val Loss: {:.8f}".format(iter_cnt, train_loss, val_loss))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = iter_cnt
            else:
                if iter_cnt - best_iteration >= early_stopping_rounds:
                    print("Early Stopping  Best Iteration: {}  "
                          "Best Val Loss: {:8f}".format(
                              best_iteration, best_val_loss))
                    break
        self.trees = trees
        self.best_iteration = best_iteration
        self.best_val_loss = best_val_loss
        print("Training Finished")

    def predict(self, X):
        return self._predict(X, self.trees[:self.best_iteration])

    def _predict(self, X, trees):
        return np.array(
                   [np.sum([tree.predict(x) for tree in trees]) for x in X])
