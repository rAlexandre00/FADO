from fado.cli.arguments.arguments import FADOArguments
from fado.runner.fl.select.base_participant_selector import ParticipantSelector
from fado.runner.fl.select.random_participant_selector import RandomParticipantSelector


class ParticipantSelectorManager:

    @classmethod
    def get_selector(cls) -> ParticipantSelector:
        args = FADOArguments()
        if args.participant_selector == 'random':
            return RandomParticipantSelector()
        elif '.py' in args.participant_selector:
            return args.get_class('participant_selector')()
        else:
            raise Exception("Specified participant selector does not exist")
