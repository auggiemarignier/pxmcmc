import numpy as np


class Outfile:
    def __init__(self, logpost, preds, chain, outpath, binary=True):
        """
        Initialises required output values, path and whether or not to save as a binary.
        """
        self.dictnry = {"logposterior": logpost, "predictions": preds, "chain": chain}
        self.outpath = outpath
        self.binary = binary

    def write_outfile(self, key):
        """
        Writes out an output value identified by key.
        """
        outfile = f"{self.outpath}/{key}"
        if self.binary:
            np.save(outfile, self.dictnry[key])
            print(f"{key} written to {outfile}.npy")
        else:
            np.savetxt(outfile, self.dictnry[key])
            print(f"{key} written to {outfile}")

    def write_outfiles(self):
        "Writes out all the output files."
        for key in self.dictnry:
            self.write_outfile(key)
