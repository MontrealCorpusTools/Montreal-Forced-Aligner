def CollectBinaries(directory):
	import shutil, os
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('dir')
	args = parser.parse_args()
	directory = args.dir
	for root, dirs, files in os.walk(directory):
		for name in files:
			ext = os.path.splitext(name)
			(key, value) = ext
			if value == '' and key != '.DS_Store' and key != 'configure' and key != 'Doxyfile' and key != 'INSTALL' and key != 'NOTES' and key != 'TODO' and key != 'Makefile':
				shutil.copy(root + '/' + name, os.getcwd())

CollectBinaries('directory')