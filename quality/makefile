PYTHON:=python

.PHONY: generate_pseudo_labels
generate_pseudo_labels: gen_data_list extract_feats gen_pseudo_labels

.PHONY: gen_data_list
gen_data_list:
	cd generate_pseudo_labels; $(PYTHON) gen_datalist.py $(dataset_root); cd ..

.PHONY: extract_feats
extract_feats:
	cd generate_pseudo_labels/extract_embedding; $(PYTHON) extract_feats.py; cd ../..

.PHONY: gen_pseudo_labels
gen_pseudo_labels:
	cd generate_pseudo_labels; $(PYTHON) gen_pseudo_labels.py; cd ..

.PHONY: train
train:
	./local_train.sh
