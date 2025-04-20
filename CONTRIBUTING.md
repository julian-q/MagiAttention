# Contributing Guide


## Git Pull Request

Please follow these steps to open a pull request:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


### Unit Tests

We use [PyTest](https://docs.pytest.org/en/latest/) to execute tests. You can install pytest by `pip install pytest`. As some of the tests require initialization of the distributed backend, GPUs are needed to execute these tests.

To set up the environment for unit testing, first change your current directory to the root directory of your local ColossalAI repository, then run
```bash
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

then, just run:

```bash
pytest tests/
```

<!-- Unit testing will be automatically conducted when you put up a pull request to the main branch. -->


### Code Style

We have some static checks when you commit your code change, please make sure you can pass all the tests and make sure the coding style meets our requirements. We use pre-commit hook to make sure the code is aligned with the writing standard. To set up the code style checking, you need to follow the steps below.

* pre-commit:
```bash
# after `pip install pre-commit` (done in step3)
# you need to set up the hooks for the first time
# which might take a while but only need to be done once
pre-commit install

# then each time before you run `git commit`
# please run the pre-commit to polish your code
pre-commit run -a

# if anything has been automatically fixed
# or required to be manually fixed by pre-commit
# please rerun `git add` to track the changes
# when everything is fixed and ready to be committed

# for more detailed information about pre-commit
# you can check: https://pre-commit.com/
```

Code format checking will be automatically executed when you commit your changes.
