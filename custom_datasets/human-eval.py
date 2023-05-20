import os
from tree_sitter import Language, Parser
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.json as js
import datasets
import json
from codetext.utils import parse_code
from codetext.parser import PythonParser

class HumanEvalConfig(datasets.BuilderConfig):
    """BuilderConfig for The Vault dataset."""

    def __init__(self, *args, datapath, **kwargs):
        """BuilderConfig for Human Eval.
        Args:
            split_set (:obj:`List[str]`): List of split set to load.
            languages (:obj:`List[str]`): List of languages to load.
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(
            *args,
            name= "human-eval",
            **kwargs,
        )
        self.datapath = datapath



class HumanEval(datasets.GeneratorBasedBuilder):
 
    BUILDER_CONFIG_CLASS = HumanEvalConfig
    BUILDER_CONFIGS = [HumanEvalConfig(datapath="human-eval")]
    # DEFAULT_CONFIG_NAME = "all-all"

    
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                                        "task_id": datasets.Value("string"),
                                        "prompt": datasets.Value("string"),
                                        "canonical_solution": datasets.Value("string"),
                                        "test": datasets.Value("string"),
                                        "docstring": datasets.Value("string"),
                                        "code": datasets.Value("string"),
                                        }),
            
        )

    def _split_generators(self, dl_manager):
        generators = []
        generators.append(datasets.SplitGenerator(
                        name='train',
                        gen_kwargs={
                            "files": [self.config.datapath],
                        },
                    ))

        return generators

    def _generate_examples(self, files):

        PY_LANGUAGE = Language('/home/namlh31/project/AI4Code/TheVault_exp/custom_datasets/tree-sitter/my-languages.so', 'python')
        parser = Parser()
        parser.set_language(PY_LANGUAGE)
        
        key = 0
        with open(self.config.datapath, "r") as f:
            data = f.readlines()
        
        for dp in data:
            if not dp.strip():
                continue
            dp = json.loads(dp)

            root = parser.parse(bytes(dp['prompt'] + dp['canonical_solution'], "utf8"))
            root_node = root.root_node

            function_list = PythonParser.get_function_list(root_node)
            docstring = ""

            if len(function_list) > 1:
                assert len(function_list) == 2
                if function_list[1].text.decode() in function_list[0].text.decode():
                    docstring = PythonParser.get_docstring(function_list[0], dp['prompt'] + dp['canonical_solution'])
                else:
                    docstring = PythonParser.get_docstring(function_list[1], dp['prompt'] + dp['canonical_solution'])
            else:
                docstring = PythonParser.get_docstring(function_list[0], dp['prompt'] + dp['canonical_solution'])

            code = root_node.text.decode().replace("\"\"\"" + docstring + "\"\"\"", "").replace("'''" + docstring + "'''", "")


            yield key, {
                        "task_id": dp["task_id"],
                        "prompt": dp["prompt"],
                        "canonical_solution": dp["canonical_solution"],
                        "test": dp["test"],
                        "docstring": docstring,
                        "code": code,
                    } 
            key += 1

            


        # for file_idx, file in enumerate(files):
        #     with open(file, 'r') as f:
        #         data = f.readlines()
            
        #     for dp in data:
        #         row = json.loads(dp)
        #         parameters = []
        #         for param in row['parameters']:
        #             parameters.append({'param': param, 'type': row['parameters'][param]})

        #         if 'docstring_params' not in row:
        #             docstring_params= {"returns": [], "raises": [], "params": [], "outlier_params": [], "others": []}
        #         else:
        #             docstring_params = reformat_docstring_params(row['docstring_params'])
        #         # print(docstring_params)
        #         yield key, {
        #                         "hexsha": row['hexsha'],
        #                         "repo": row['repo'],
        #                         "path": row['path'], 
        #                         "license": row['license'], 
        #                         "language": row['language'],
        #                         "identifier": row['identifier'],
        #                         "return_type": row['return_type'],
        #                         "original_string": row['original_string'][0],
        #                         "original_docstring": row['original_docstring'],
        #                         "docstring": row['docstring'],
        #                         "docstring_tokens": row['docstring_tokens'],
        #                         "code": row['code'],
        #                         "code_tokens": row['code_tokens'],
        #                         "short_docstring": row['short_docstring'],
        #                         "short_docstring_tokens": row['short_docstring_tokens'],
        #                         "comment": row['comment'],
        #                         "parameters": parameters,
        #                         "docstring_params": docstring_params
        #                     } 
        #         key += 1

