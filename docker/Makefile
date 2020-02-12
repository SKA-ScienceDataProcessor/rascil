docker:
	cd rascil-no-data;docker build -t timcornwell/rascil-no-data .
	cd rascil-full;docker build -t timcornwell/rascil-full .
	cd rascil-notebook;docker build -t timcornwell/rascil-notebook .

docker-no-cache:
	cd rascil-no-data;docker build --no-cache -t timcornwell/rascil-no-data .
	cd rascil-full;docker build --no-cache -t timcornwell/rascil-full .
	cd rascil-notebook;docker build --no-cache -t timcornwell/rascil-notebook .

docker-push:
	docker push timcornwell/rascil-no-data
	docker push timcornwell/rascil-full
	docker push timcornwell/rascil-notebook
