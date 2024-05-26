import numpy as np
import pandas as pd
import javalang
import os
from collections import Counter
import statistics
import re


def feature_dictionary(f_dict):
        result_dict = {}
        for d in f_dict:
            for key, value in d.items():
                if key in result_dict:
                    result_dict[key].append(value)
                else:
                    result_dict[key] = [value]
        return result_dict


class Parsing_and_Feature_Extraction:

    def __init__(self, node):
        self.node = node

    def node_name(self):
        return self.node.name
    def methods_count(self):
        method_names = [m.name for m in self.node.methods]
        return len(method_names)
    def fields_count(self):
        return len(self.node.fields)
    def pub_methods_invoc_count(self):
        pub_methods = [m.name for m in self.node.methods if m.modifiers == {'public'}]
        inv_cnt = [invocation.member for _, invocation in self.node.filter(javalang.tree.MethodInvocation)]
        return len(pub_methods) + len(inv_cnt)
    def imp_interfaces_count(self):
        imp_inter = [imp_inter.name for _, imp_inter in self.node.filter(javalang.tree.InterfaceDeclaration)]
        return len(imp_inter)
    def statements_count(self):
        in_each_method_statements = []
        for m in self.node.methods:
            statements = [stat for _, stat in m.filter(javalang.tree.Statement) if not isinstance(stat, javalang.tree.BlockStatement)]
            in_each_method_statements.append(len(statements))
        try:
            return max(in_each_method_statements)
        except:
            return 0
    def conditional_and_loop_statements_count(self):
        in_each_method_statements_con_loop = []
        for m in self.node.methods:
            cond_stat = [cond for _, cond in m.filter(javalang.tree.Statement) if isinstance(cond, javalang.tree.IfStatement) or isinstance(cond, javalang.tree.SwitchStatement) or isinstance(cond, javalang.tree.WhileStatement) or isinstance(cond, javalang.tree.DoStatement) or isinstance(cond, javalang.tree.ForStatement)]
            in_each_method_statements_con_loop.append(len(cond_stat))
        try:
            return max(in_each_method_statements_con_loop)
        except:
            return 0
    def exceptions_in_throws_clause_count(self):
        in_each_method_exceptions = []
        for m in self.node.methods:
            ex_throws = [throws.expression for _, throws in m.filter(javalang.tree.ThrowStatement)]
            in_each_method_exceptions.append(len(ex_throws))
        try:
            return max(in_each_method_exceptions)
        except:
            return 0
    def return_statements_count(self):  
        in_each_method_return_statements = []
        for m in self.node.methods:
            ret_stat = [ret.expression for _, ret in m.filter(javalang.tree.ReturnStatement) if ret.expression is not None]
            in_each_method_return_statements.append(len(ret_stat))
        try:
            return max(in_each_method_return_statements)
        except:
            return 0
    def block_comments_count(self):
        block_stat = [block for _, block in self.node.filter(javalang.tree.Documented) if block.documentation is not None]
        return len(block_stat)
    def avg_method_length(self):
        method_names = [m.name for m in self.node.methods]
        try:
            return statistics.mean([len(method) for method in method_names])
        except:
            return 0
    def block_comments_length(self):
        length = 0
        for _, doc in self.node.filter(javalang.tree.Documented):
            if doc.documentation is not None:
                words = re.findall(r'\w+', doc.documentation)
                word_count = len(words)
                length += word_count
        return length
    def words_in_comments_statements(self):
        try:
            return self.block_comments_length() / self.statements_count()
        except ZeroDivisionError:
            return 0
        # return self.block_comments_length() / self.statements_count()
    
    def extract_features(self):
        return {
            'class_name': self.node_name(),
            'MTH': self.methods_count(),
            'FLD': self.fields_count(),
            'RFC': self.pub_methods_invoc_count(),
            'INT': self.imp_interfaces_count(),
            'SZ': self.statements_count(),
            'CPX': self.conditional_and_loop_statements_count(),
            'EX': self.exceptions_in_throws_clause_count(),
            'RET': self.return_statements_count(),
            'BCM': self.block_comments_count(),
            'NML': self.avg_method_length(),
            'WRD': self.block_comments_length(),
            'DCM': self.words_in_comments_statements()
        }
        
        
if __name__ == '__main__':
    feature_list = []
    for (dirpath, dirnames, filenames) in os.walk("./resources"):
        # print('dirpath: ', dirpath)
        if 'src/com/google/javascript/jscomp' in dirpath:
            for filename in filenames:
                if filename.endswith('.java'):
                    with open(os.path.join(dirpath, filename), 'r') as file:
                        tree = javalang.parse.parse(file.read())
                        for path, node in tree.filter(javalang.tree.ClassDeclaration):
                            if node.name == filename.split('/')[-1].split('.')[0]:
                                parser = Parsing_and_Feature_Extraction(node)
                                feature_list.append(parser.extract_features())
    
    feature_result = feature_dictionary(feature_list)        

    feature_frame = pd.DataFrame.from_dict(feature_result)

    feature_frame.to_csv('results/feature_vectors.csv', index=False)
    print('Feature vectors extracted successfully! Check the file feature_vectors.csv in the results folder.')
           