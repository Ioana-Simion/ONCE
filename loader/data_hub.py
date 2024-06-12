from typing import Union

from oba import Obj
from process.mind.unidep import UniDep

from loader.depot.depot_hub import DepotHub


class DataHub:
    def __init__(
        self,
        depot: Union[UniDep, str],
        order,
        append=None,
    ):
        self.depot = depot if isinstance(depot, UniDep) else DepotHub.get(depot)
        self.order = Obj.raw(order)
        self.append = Obj.raw(append) or []

        # self.depot.select_cols(self.order + self.append)
