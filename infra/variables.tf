variable "project_id" {
  type = string
}

variable "region" {
  type    = string
  default = "europe-west1"
}

variable "service_name" {
  type    = string
  default = "odyn-simulator"
}

variable "image" {
  type = string
}

variable "results_bucket_name" {
  type = string
}

variable "results_prefix" {
  type    = string
  default = "simulations"
}

variable "allowed_invoker" {
  type        = string
  description = "Principal allowed to invoke Cloud Run. Example: group:engineering@odyn.ai"
}

variable "min_instances" {
  type    = number
  default = 1
}

variable "max_instances" {
  type    = number
  default = 10
}
