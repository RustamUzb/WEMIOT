from configparser import ConfigParser
import os


def createDefaultConfig():
    # Get the configparser object
    config_object = ConfigParser()

    config_object["FOLDERS"] = {
        "OUTP_FILES": os.path.join(os.path.dirname(os.path.dirname(__file__)), 'files')
    }

    config_object['DEFAULTS'] = {
        'printResults': 'console'
    }

    # Write the above sections to config.ini file
    with open('../conf/config.ini', 'w') as conf:
        config_object.write(conf)