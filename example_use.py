import sys
import os
# Agent
from tatk.dialog_agent import PipelineAgent, BiSession
from tatk.nlu.svm.multiwoz import SVMNLU
from tatk.nlu.bert.multiwoz import BERTNLU
from tatk.dst.rule.multiwoz import RuleDST
from tatk.policy.hdsa.multiwoz import HDSA
from tatk.policy.rule.multiwoz import Rule
from tatk.policy.larl.multiwoz import LaRLPolicy
from tatk.nlg.template.multiwoz import TemplateNLG
from tatk.evaluator.multiwoz_eval import MultiWozEvaluator
from pprint import pprint

import random
import numpy as np


# NLU
# simpleBert
sys_nlu = BERTNLU(mode='usr')
# DST
sys_dst = RuleDST()
# POLICY
sys_policy = LaRLPolicy(
    archive_file="/home/mawenchang/TATK/tatk/policy/larl/multiwoz/data/larl_model.zip")
#sys_policy = HDSA()
sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, None, 'sys')

# svm nlu trained on usr sentence of multiwoz
# go to README.md under `tatk/tatk/nlu/svm/multiwoz` for more information
"""sys_nlu = BERTNLU(mode='usr')
# simple rule DST
sys_dst = RuleDST()
# rule policy
sys_policy = Rule(character='sys')
# template NLG
sys_nlg = TemplateNLG(is_user=False)
# assemble
sys_agent = PipelineAgent(sys_nlu, sys_dst, sys_policy, sys_nlg, 'sys')"""

# # bert nlu trained on sys sentence of multiwoz
# # go to README.md under `tatk/tatk/nlu/bert/multiwoz` for more information
user_nlu = BERTNLU(mode='sys')
# # not use dst
#user_dst = RuleDST()
# # rule policy
user_policy = Rule(character='usr')
# # template NLG
user_nlg = TemplateNLG(is_user=True)
# # assemble
user_agent = PipelineAgent(user_nlu, None, user_policy, user_nlg, "user")

evaluator = MultiWozEvaluator()
sess = BiSession(sys_agent=sys_agent, user_agent=user_agent,
                 kb_query=None, evaluator=evaluator)

random.seed(19990701)
np.random.seed(19990701)
sys_response = ''
total_dialog = 100
precision = 0
recall = 0
f1 = 0
suc_num = 0
for j in range(total_dialog):
    sess.init_session()
    print('init goal:')
    pprint(sess.evaluator.goal)
    print('-'*50)
    for i in range(40):
        sys_response, user_response, session_over, reward = sess.next_turn(
            sys_response)
        print('user:', user_response)
        print('sys:', sys_response)
        if session_over is True:
            if sess.evaluator.task_success() == 1:
                suc_num = suc_num+1
            print('task success:', sess.evaluator.task_success())
            print('book rate:', sess.evaluator.book_rate())
            print('inform precision/recall/f1:', sess.evaluator.inform_F1())
            stats = sess.evaluator.inform_F1()
            if(stats[0] != None):
                precision = precision+stats[0]
            if(stats[1] != None):
                recall = recall+stats[1]
            if(stats[2] != None):
                f1 = f1+stats[2]
            else:
                suc_num = suc_num-1
            print('-'*50)
            print('final goal:')
            pprint(sess.evaluator.goal)
            print('='*100)
            break
print("success number of dialogs/tot:", suc_num/total_dialog)
print("average precision:", precision/total_dialog)
print("average recall:", recall/total_dialog)
print("average f1:", f1/total_dialog)
