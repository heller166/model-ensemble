# Linear Model Ensemble
Spring 2022

Code author
-----------
Grannie Casey, Juan Francisco Alfaro

Installation
------------
These components are installed:
- JDK 1.8
- Scala 2.12.15
- Hadoop 3.3.1
- Spark 3.2.1
- Maven
- AWS CLI (for EMR execution)

Environment
-----------
1) Example ~/.bash_aliases:  
   export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64  
   export SCALA_HOME=/home/juan/tools/scala/scala-2.11.15  
   export SPARK_HOME=/home/juan/tools/spark/spark-2.3.1-bin-without-hadoop  
   export HADOOP_HOME=/home/juan/hadoop-3.3.1  
   export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop  
   export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$JAVA_HOME/bin
   export SPARK_DIST_CLASSPATH=$(hadoop classpath)


2) Explicitly set JAVA_HOME in $HADOOP_HOME/etc/hadoop/hadoop-env.sh:    
   export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

Execution
---------
All of the build & execution commands are organized in the Makefile.
1) Unzip project file.
2) Open command prompt.
3) Navigate to directory where project files unzipped.
4) Edit the Makefile to customize the environment at the top.  
   Sufficient for standalone: hadoop.root, jar.name, local.input  
   Other defaults acceptable for running standalone.
5) Go to this [link](https://drive.google.com/file/d/1uPufGONV9_Crb16bewI00Tyz6fb_HQl2/view?usp=sharing), download data and put it in an input directory
6) Standalone Hadoop
    - make switch-standalone		-- set standalone Hadoop environment (execute once)
    - make local local.numModels=\<int>
7) AWS EMR Hadoop: (you must configure the emr.* config parameters at top of Makefile)
   - make upload-input-aws  -- only before first execution
   - make aws aws.numModels=\<int> 
   - download-output-aws			-- after successful
8) R2 Score will be displayed on stdout