from classification import ClassificationModel
from argumentparser import *

class CatBoost(ClassificationModel):
    def __init__(self, _args):
        self.args = _args

    def computeModel(XTrain, yTrain, max_depth, n_estimators, learning_rate):
        from catboost import CatBoostClassifier

        classifier = CatBoostClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            verbose=0,
            auto_class_weights='Balanced'
        )
        classifier.fit(XTrain, yTrain)
        return classifier

    def compute(self):
        import timeit
        start = timeit.default_timer()

        XTrain, XTest, yTrain, yTest = ClassificationModel.preprocessData(self.args, True)

        classifier = CatBoost.computeModel(XTrain, yTrain, self.args.max_depth, self.args.n_estimators, self.args.learning_rate)
        yPred = ClassificationModel.predictModel(classifier, XTest)
        confusionMatrix = ClassificationModel.getConfusionMatrix(yPred, yTest)
        rocCurve = ClassificationModel.getRocCurve(yPred, yTest)

        if self.args.print_accuracy:
            print(confusionMatrix, ClassificationModel.getAccuracy(confusionMatrix))

        stop = timeit.default_timer()
        return confusionMatrix, rocCurve, ClassificationModel.getAccuracy(confusionMatrix), stop - start, classifier

    def computeCrossValidation(self):
        from sklearn.model_selection import cross_validate

        X, y = ClassificationModel.preprocessDataCrossValidation(self.args, True)
        classifier = CatBoost.computeModel(X, y, self.args.max_depth, self.args.n_estimators, self.args.learning_rate)

        cv_results = cross_validate(classifier, X, y, cv=self.args.k_fold_cross_validation)

        if self.args.print_accuracy:
            print(cv_results)

        return cv_results
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.setBasicArguments()
    parser.setCatBoostArguments()
    args = parser.getArguments()

    model = CatBoost(args)

    if args.cross_validation:
        model.computeCrossValidation()
    else:
        model.compute()
