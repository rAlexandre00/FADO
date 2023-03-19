from typing import Type

from fado.cli.arguments.arguments import FADOArguments
from fado.security.defend.server.clipping_defender import ClippingDefender

from fado.security.defend.server.server_defense_base import ServerDefender


class ServerDefenseManager:

    @classmethod
    def get_defender(cls) -> ServerDefender:
        args = FADOArguments()

        server_defense_name = args.server_defense_name if 'server_defense_name' in args else None
        
        if server_defense_name == 'clip':
            return ClippingDefender()
        else:
            return ServerDefender()
