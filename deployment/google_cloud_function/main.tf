
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "3.5.0"
    }
  }
}

provider "google" {
  credentials = file("~/credentials/deployment-316521-6af7eb229d43.json")

  project = "deployment-316521"
  region  = "europe-west3"
  #zone    = "us-central1-c"
}

#resource "google_compute_network" "vpc_network" {
#  name = "terraform-network"
#}


resource "google_storage_bucket" "bucket" {
  name = "mltemplate-cloudfunction"
}

resource "google_storage_bucket_object" "archive" {
  name   = "function_deployment.zip"
  bucket = google_storage_bucket.bucket.name
  source = "../../output/function_deployment/function_deployment.zip"
}

resource "google_cloudfunctions_function" "function" {
  name        = "api_predict"
  description = "Testing of mltemplate with cloud function"
  runtime     = "python37"

  available_memory_mb   = 512
  source_archive_bucket = google_storage_bucket.bucket.name
  source_archive_object = google_storage_bucket_object.archive.name
  trigger_http          = true
  entry_point           = "predict"

  timeouts {
    create = "20m"
  }

}

//resource "aws_db_instance" "example" {
//  # ...
//
//  timeouts {
//    create = "60m"
//    delete = "2h"
//  }
//}

# providing invocation permissions does not work somehow... below to alternative ways
# maybe it has to do with me deleting one of these initial permissions back then?

# IAM entry for all users to invoke the function
#resource "google_cloudfunctions_function_iam_member" "invoker" {
#  project        = google_cloudfunctions_function.function.project
#  region         = google_cloudfunctions_function.function.region
#  cloud_function = google_cloudfunctions_function.function.name

#  role   = "roles/cloudfunctions.invoker"
#  member = "allUsers"
#}

#resource "google_cloudfunctions_function_iam_binding" "binding" {
#  project = google_cloudfunctions_function.function.project
#  region = google_cloudfunctions_function.function.region
#  cloud_function = google_cloudfunctions_function.function.name
#  role = "roles/cloudfunctions.invoker"
#  members = [
#    "allUsers",
#  ]
#}