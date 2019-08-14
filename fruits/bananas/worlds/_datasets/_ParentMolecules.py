import mandalka

from .. import StorageWorld, Table

@mandalka.node
class ParentMolecules(StorageWorld):

    """
    Replace uid with parent_uid and smiles with parent_smiles,
    if they exist.
    """

    def build(self, source):
        self.data = source.data.slice[:]
        self.data["child_uid"] = Table(self.data["uid"])
        self.data["child_smiles"] = Table(self.data["smiles"])

        replace_idx = source.data["parent_uid"] != ""

        _uid = self.data["uid"]
        _uid[replace_idx] = self.data["parent_uid"][replace_idx]
        self.data["uid"] = self.data.get_container("uid")(_uid)
        del self.data["parent_uid"]

        _smiles = self.data["smiles"]
        _smiles[replace_idx] = self.data["parent_smiles"][replace_idx]
        self.data["smiles"] = self.data.get_container("smiles")(_smiles)
        del self.data["parent_smiles"]
