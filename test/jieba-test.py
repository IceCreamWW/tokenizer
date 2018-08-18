import jieba

print('\t'.join(jieba.cut('杨殿富于１９９７年４月２８日被提拔为市公安局预审处处长。', cut_all=False)))
print('\t'.join(jieba.cut_for_search('杨殿富于１９９７年４月２８日被提拔为市公安局预审处处长。')))
