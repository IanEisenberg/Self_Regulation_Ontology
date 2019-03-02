define setup
expfactory_token:/dev/null
results_directory:/results
base_directory:/workdir/Self_Regulation_Ontology
dataset:Complete_10-08-2017
endef
export setup

setup-docker:
	@echo  "$$setup">Self_Regulation_Settings.txt
	python setup.py install 

docker-clean:
	-sudo rm ~/singularity/*
	-docker stop $(docker ps -a -q)
	-docker rm $(docker ps -a -q)
	-docker images -q --filter dangling=true | xargs docker rmi -f

docker2singularity:
	docker run -v /var/run/docker.sock:/var/run/docker.sock -v /home/poldrack/singularity:/output --privileged -t --rm singularityware/docker2singularity selfregulation

# this adds a character to the RUN command
# # which causes all commands below to be re-excuted
docker-build:
	cat Dockerfile | sed 's/RUN echo "x/RUN echo "xx/'>Dockerfile.tmp
	cp Dockerfile.tmp Dockerfile
	docker build -t selfregulation .

copy:
	scp ~/singularity/selfregulation-*.img wrangler.tacc.utexas.edu:/work/01329/poldrack/stampede2/singularity_images

shell:
	docker run -v /tmp/results:/SRO/results -it --entrypoint=bash selfregulation 

