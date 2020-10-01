.PHONY : test

test :
	python -m pytest --benchmark-skip

benchmark :
	python -m pytest --benchmark-only

benchmark_record :
	python -m pytest --benchmark-only --benchmark-autosave
