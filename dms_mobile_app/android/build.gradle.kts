allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}

subprojects {
    val configureAndroid: Project.() -> Unit = {
        val androidExt = extensions.findByName("android")
        if (androidExt != null) {
            try {
                val getNamespace = androidExt.javaClass.getMethod("getNamespace")
                val namespace = getNamespace.invoke(androidExt)
                if (namespace == null) {
                    val setNamespace = androidExt.javaClass.getMethod("setNamespace", String::class.java)
                    val generatedNamespace = "com.dms.mobile." + name.replace("-", "_").replace(":", "_")
                    setNamespace.invoke(androidExt, generatedNamespace)
                    logger.info("Dynamically set namespace to $generatedNamespace for project $name")
                }
            } catch (e: Exception) {
                // Ignore if method does not exist
            }
        }
    }

    if (state.executed) {
        configureAndroid()
    } else {
        afterEvaluate {
            configureAndroid()
        }
    }
}
