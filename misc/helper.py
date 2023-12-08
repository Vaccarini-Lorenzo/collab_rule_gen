import re


class Helper:
    @staticmethod
    def check_in_dict(dictionary, *args):
        for argument in args:
            if not argument in dictionary:
                print("Error parsing json")
                exit(1)

    @staticmethod
    def collapse_whitespace(text):
        if len(text) == 0:
            return text
        new_text = re.sub(r'\s+', ' ', text)
        if new_text[0] == " ":
            new_text = new_text[1:]
        if new_text[len(new_text) - 1] == " ":
            new_text = new_text[:-1]
        return new_text

    @staticmethod
    def sort_state_name(state):
        words = state.split(" ")

        # Sort the list of words alphabetically
        words.sort()

        # Join the sorted list of words back into a string
        return " ".join(words)