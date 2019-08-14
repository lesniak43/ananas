import mandalka

from .. import StorageWorld, Variable


@mandalka.node
class FullYearSplit:
    def __init__(self, *, source):
        self.source = source
    def get_splits(self):
        return [YearSplit(source=self.source, test_year=int(year)) for year in sorted(set(self.source.data["year"]))[1:]]

@mandalka.node
class YearSplit(StorageWorld):

    def build(self, *, source, test_year):
        self.data = source.data.slice[:]
        del self.data["year_split_test_year"]
        self.data["year_split_test_year"] = Variable(test_year)

    @mandalka.lazy
    def get_train(self):
        return YearSplitInner(
            split=self, which="train")

    @mandalka.lazy
    def get_test(self):
        return YearSplitInner(
            split=self, which="test")

@mandalka.node
class YearSplitInner(StorageWorld):

    def build(self, *, split, which):
        if which == "train":
            idx = split.data["year"] < split.data["year_split_test_year"]
        elif which == "test":
            idx = split.data["year"] == split.data["year_split_test_year"]
        else:
            raise ValueError("'which' must be 'train' or 'test', not {}".format(which))
        self.data = split.data.slice[idx]
        del self.data["year_split_test_year"]
