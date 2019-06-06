"""Text Processing for English"""


class EnglishText():
    """English Text Processing
    """
    def __init__(self):
        """Constructor
        """
        # symbols
        _pad = "_PAD_"
        _eos = "_EOS_"
        _unk = "_UNK_"
        _characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? "

        symbols = [_pad] + [_eos] + [_unk] + list(_characters)

        self.symbol_to_id = {symbol: idx for idx, symbol in enumerate(symbols)}
        self.id_to_symbol = {idx: symbol for idx, symbol in enumerate(symbols)}

    def text_to_sequence(self, textpath):
        """Reads text from file and returns it a sequence of ids
            Args:
                textpath: Full path to the text file on disk
                returns: list of integers, where each integer corresponds to a character as specified in
                         symbol_to_id mapping
        """
        # read contents of text file
        with open(textpath, "r") as fp:
            text = fp.readlines()
            text = " ".join(text).strip("\n")

        # convert text to a sequence of ids
        sequence = [self.symbol_to_id[char] if char in self.symbol_to_id else self.symbol_to_id["_UNK_"]
                    for char in text]

        return sequence
