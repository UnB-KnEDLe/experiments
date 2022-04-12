from multiprocessing import cpu_count
import pandas as pd
import subprocess
from metaflow import FlowSpec, step, Parameter
from pre_structure import LabelStructure


class PreProcessFlow(FlowSpec):
    """
    A flow to pre-process data for the SegmentationFlow flow.
    """

    act = Parameter(
        "act",
        help="Type of act to process (example: 'aviso de licitação'",
        required=True,
        type=str,
    )

    labeled = Parameter(
        "labeled",
        help="Path to the dataset with labeled acts samples",
        required=True,
        type=str,
    )

    dodf_dir = Parameter(
        "dodf_dir",
        help="Path to the directory with dodfs txts",
        required=True,
        type=str,
    )

    output = Parameter(
        "output", help="Path to store the output dataset", required=True, type=str
    )

    @step
    def start(self):
        self.dodfs_path = subprocess.Popen(
            f'find "{self.dodf_dir}" -type f -name "*.txt"',
            shell=True,
            stdout=subprocess.PIPE,
        ).stdout.read()
        self.next(self.create_paths_dataset)

    @step
    def create_paths_dataset(self):
        """
        Creates a dataset containing the paths for the DODF's text files, removing files of section
        I and section III. The dataset contains the following columns, which uniquely identify each
        text file:
            dodf_path, edition_date, dodf_number, dodf_date, section

        It generates the attributes 'paths_df' and 'dodf_files', which are the dataset and a list of
        the unique entries in the dataset, respectively.
        """

        paths_df = pd.DataFrame(
            {"dodf_path": self.dodfs_path.decode("utf-8").splitlines()}
        )
        paths_df["edition_date"] = paths_df.dodf_path.str.extract(
            pat="(DODF\s\d{3}\s\d{2}-\d{2}-\d{4}\s\d|DODF\s\d{3}\s\d{2}-\d{2}-\d{4}|"
            "DODF\s\d{2}\s\d{2}-\d{2}-\d{4}|DODF\s\d{2}\s\d{2}-\d{2}-\d{4}\s\d|DODF\s\d{3}\s\d{2}"
            "-\d{2}-\d{2}\s\d|DODF\s\d{2}\s\d{2}-\d{2}-\d{2}\s\d|DODF\s\d{3}\s\d{2}-\d{2}-\d{2}|"
            "DODF\s\d{2}\s\d{2}-\d{2}-\d{2}|DODF\s\d{3}\s\d{2}-\d{2}\s\d{4}\s\d|DODF\s\d{3}\s\d{2}"
            "-\d{2}-\d{4}\s\d)"
        )
        paths_df["edition_date"] = paths_df["edition_date"].str.replace(r"DODF\s", "")

        paths_df[["dodf_number", "dodf_date"]] = paths_df["edition_date"].str.split(
            " ", 1, expand=True
        )
        paths_df["section"] = paths_df.dodf_path.str.extract(pat="(\s\d.txt)")
        paths_df["section"] = paths_df["section"].str.replace(r".txt", "").astype(str)
        paths_df["dodf_date"] = (
            paths_df["dodf_date"].str.replace(r"-|/", ".").astype(str)
        )
        paths_df = paths_df.loc[paths_df["section"] != "1"]
        paths_df = paths_df.loc[paths_df["section"] != "2"]
        paths_df["dodf_date"] = (
            paths_df["dodf_date"].str.replace(r"\s\d$", "").astype(str)
        )

        self.paths_df = paths_df

        dodf_files = list(set(paths_df.dodf_path.to_list()))
        size = len(dodf_files) // cpu_count()
        self.chunks = [
            dodf_files[i : i + size] for i in range(0, len(dodf_files), size)
        ]

        self.next(self.load_acts)

    @step
    def load_acts(self):
        """
        Loads and clears an annotated act dataset (not necessarily acts of type self.act).
        This dataset must contain at least the columns 'text', 'ato', 'dodf'.

        Generates the attribute self.acts, which is the dataset filtered by the target act type.
        """
        acts_df = pd.read_parquet(self.labeled)
        acts_df.drop(["arquivo_rast"], axis=1, inplace=True)
        acts_df.drop_duplicates(inplace=True)

        def fix_date(s):
            if s == "nan":
                return "nan"
            if s == "NaN":
                return "nan"

            s = str(s).split("_")
            new_date = s[1] + " " + s[2]
            return new_date

        acts_df["dodf"] = acts_df["dodf"].apply(fix_date)
        self.acts = acts_df[acts_df["ato"] == self.act]
        print(self.act, self.acts.shape[0])

        self.next(self.create_iob, foreach="chunks")

    @step
    def create_iob(self):
        """
        For a chunk of DODFs, it separates the text into blocks, tokenizes each block into sentences,
        and creates the sentences' IOB tags.

        It generates the attribute self.iobs with the IOB sentences separated by newlines and
        blocks of sentences separated by two newlines.
        """
        self.iobs = ""
        for path in self.input:
            list_position = []
            if path != "nan":
                edicao = self.paths_df[self.paths_df["dodf_path"] == path][
                    "edition_date"
                ].values[0]
                acts_in_dodf = self.acts[self.acts["dodf"] == edicao]
                segments = acts_in_dodf.text.tolist()
                if len(segments) > 0:
                    for s, segment in enumerate(segments):
                        ls = LabelStructure()
                        try:
                            aligment = ls.find_alignment(path, segment)
                        except ValueError:
                            aligment = []

                        if len(aligment) != 0:
                            init = aligment.aligned[0][0][0]
                            end = aligment.aligned[0][0][1]
                            list_position.append((s, segment, init, end))

                    text_segmentado = ls.segmentor(path, list_position)
                    iob = ls.sentence_labeling(text_segmentado)

                    self.iobs += iob

        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Writes the result of the previous steps to the output file.
        """
        with open(self.output + "segmentacao_" + self.act + ".iob", "a+") as f_out:
            for branch in inputs:
                f_out.write(branch.iobs)

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    PreProcessFlow()
