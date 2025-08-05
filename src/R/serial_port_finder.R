find_usb_port <- function() {
  sysname <- Sys.info()[["sysname"]]

  if (sysname == "Windows") {
    ports <- try(serial::listPorts(), silent = TRUE)
    if (inherits(ports, "try-error") || length(ports) == 0) {
      stop("No serial ports found on Windows")
    }
    usb_ports <- ports[grepl("usb", ports, ignore.case = TRUE)]
    if (length(usb_ports) == 0) {
      stop("No USB serial port found on Windows")
    }
    cat("Found USB port on Windows:", usb_ports[1], "\n")
    return(usb_ports[1])

  } else if (sysname %in% c("Darwin", "Linux")) {
    usb_ports <- Sys.glob("/dev/tty.usbmodem*")
    if (length(usb_ports) == 0) {
      usb_ports <- Sys.glob("/dev/ttyUSB*")
    }
    if (length(usb_ports) == 0) {
      stop("No /dev/tty.usbmodem* or /dev/ttyUSB* device found on Unix/macOS/Linux")
    }
    cat("Found USB port on Unix/macOS/Linux:", usb_ports[1], "\n")
    return(usb_ports[1])

  } else {
    stop("Unsupported OS for USB port detection")
  }
}
