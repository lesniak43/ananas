import numpy as np

from herbivores import align_columns, merge_columns
import mandalka

from . import StorageWorld, ProxyWorld, Variable, Header, Table, ProxyContainer
from ._kernels._tanimoto import tanimoto_minmax_similarity

@mandalka.node
class FingerprintRepresentationMaker(ProxyWorld):
    def build_proxy(self, *, fingerprint):
        self.data["keys"] = ProxyContainer(
            (lambda : fingerprint.data[("fingerprint", "keys")]),
            Header,
        )
    @mandalka.lazy
    def __call__(self, *, fingerprint):
        return FingerprintRepresentation(
            fingerprint=fingerprint,
            maker=self,
        )
@mandalka.node
class FingerprintRepresentation(ProxyWorld):
    def build_proxy(self, *, fingerprint, maker):
        self.data = fingerprint.data.proxy
        del self.data["representation"]
        def f():
            representation = align_columns(
                fingerprint.data[("fingerprint", "data")],
                fingerprint.data[("fingerprint", "keys")],
                maker.data["keys"],
                fill_value=0.,
            )
            if hasattr(representation, "toarray"):
                representation = representation.toarray()
            representation = representation.astype(np.float32)
            return representation
        self.data["representation"] = ProxyContainer(f, Table)

@mandalka.node
class TanimotoMinMaxRepresentationMaker(ProxyWorld):
    def build_proxy(self, *, fingerprint):
        self.data["data"] = ProxyContainer(
            (lambda : fingerprint.data[("fingerprint", "data")]),
            Header,
        )
        self.data["keys"] = ProxyContainer(
            (lambda : fingerprint.data[("fingerprint", "keys")]),
            Header,
        )
    @mandalka.lazy
    def __call__(self, *, fingerprint):
        return TanimotoMinMaxRepresentation(
            fingerprint=fingerprint,
            maker=self,
        )
@mandalka.node
class TanimotoMinMaxRepresentation(ProxyWorld):
    def build_proxy(self, *, fingerprint, maker):
        self.data = fingerprint.data.proxy
        del self.data["representation"]
        def f():
            return tanimoto_minmax_similarity(*merge_columns(
                fingerprint.data[("fingerprint", "data")],
                fingerprint.data[("fingerprint", "keys")],
                maker.data["data"],
                maker.data["keys"],
                fill_value=0.,
            )[:2])
        self.data["representation"] = ProxyContainer(f, Table)

@mandalka.node
class RepresentationMeanNormalizer(StorageWorld):
    def build(self, *, source):
        arr = source.data["representation"].copy()
        assert isinstance(arr, np.ndarray), "currently only dense representation allowed"
        mean = np.nanmean(arr, axis=0).reshape(1,-1)
        mean[np.isnan(mean)] = 0. # nan columns
        arr -= mean
        arr[np.isnan(arr)] = 0.
        variance = np.maximum(np.var(arr, axis=0), 1e-8).reshape(1,-1)
        self.data["mean"] = Header(mean)
        self.data["variance"] = Header(variance)
    @mandalka.lazy
    def __call__(self, *, source):
        return RepresentationMeanNormalized(
            source=source,
            normalizer=self,
        )
@mandalka.node
class RepresentationMeanNormalized(ProxyWorld):
    def build_proxy(self, *, source, normalizer):
        self.data = source.data.proxy
        del self.data["representation"]
        def f():
            arr = np.array(source.data["representation"].copy())
            arr -= normalizer.data["mean"]
            arr[np.isnan(arr)] = 0.
            arr /= normalizer.data["variance"]
            np.clip(arr, -5., 5., out=arr)
            return arr
        self.data["representation"] = ProxyContainer(f, Table)

@mandalka.node
class RepresentationMedianNormalizer(StorageWorld):
    def build(self, *, source):
        arr = source.data["representation"].copy()
        assert isinstance(arr, np.ndarray), "currently only dense representation allowed"
        median = np.nanmedian(arr, axis=0).reshape(1,-1)
        median[np.isnan(median)] = 0. # nan columns
        arr -= median
        arr[np.isnan(arr)] = 0.
        idx = (arr.shape[0] * 84) // 100
        variance = np.maximum(np.square(np.sort(arr, axis=0)[idx]), 1e-8).reshape(1,-1)
        self.data["median"] = Header(median)
        self.data["variance"] = Header(variance)
    @mandalka.lazy
    def __call__(self, *, source):
        return RepresentationMedianNormalized(
            source=source,
            normalizer=self,
        )
@mandalka.node
class RepresentationMedianNormalized(ProxyWorld):
    def build_proxy(self, *, source, normalizer):
        self.data = source.data.proxy
        del self.data["representation"]
        def f():
            arr = np.array(source.data["representation"].copy())
            arr -= normalizer.data["median"]
            arr[np.isnan(arr)] = 0.
            arr /= normalizer.data["variance"]
            np.clip(arr, -5., 5., out=arr)
            return arr
        self.data["representation"] = ProxyContainer(f, Table)

@mandalka.node
class SortedRowsRepresentation(ProxyWorld):
    def build_proxy(self, *, source):
        self.data = source.data.proxy
        def f():
            return np.sort(source.data["representation"], axis=1)
        self.data["representation"] = ProxyContainer(f, Table)
