from json import dumps


class Printable:
    def to_json(self):
        return dumps(self, default=lambda o: o.__dict__,
                     sort_keys=True, indent=4)