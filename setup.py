from setuptools import setup, find_packages


setup(name="MLHelper",
      version="0.11.4",
      description="Collection of useful helper methods, mostly for manipulating arrays.",
      url="https://github.com/Daniel451/MLHelper",
      author="Daniel Speck",
      author_email="daniel451@mailbox.org",
      license="None",
      packages=find_packages(exclude=["contrib", "docs", "build", "dist", "MLHelper.egg-info", "tests*"]),
      install_requires=["numpy", "scipy"],
      python_requires=">=3.6",
      zip_safe=False)
