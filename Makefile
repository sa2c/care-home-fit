PARAM_SETS := full medium

.PHONY : test clean clean_timings

test :
	python -m pytest --benchmark-skip

sample_data :
	-mkdir sampledata
	for set in ${PARAM_SETS}; do \
	    python generate_sample_data.py $${set} --output_dir=sampledata ; \
	done

benchmark :
	python -m pytest --benchmark-only

benchmark_record :
	python -m pytest --benchmark-only --benchmark-autosave

clean :
	-for set in ${PARAM_SETS}; do \
	    rm sampledata/*$${set}.csv ; \
	done
	-rmdir sampledata

clean_timings :
	-rm -r .benchmarks

clean_all : clean clean_timings
