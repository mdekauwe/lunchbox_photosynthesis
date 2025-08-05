library(shiny)
library(reticulate)
library(ggplot2)

use_python("/opt/local/bin/python", required = TRUE)

# Source Python modules
source_python("/Users/xj21307/src/python/lunchbox_photosynthesis/src/python/serial_port_finder.py")
source_python("/Users/xj21307/src/python/lunchbox_photosynthesis/src/python/xensiv_pas_co2_sensor.py")
source_python("/Users/xj21307/src/python/lunchbox_photosynthesis/src/python/lunchbox_logger.py")

port <- find_usb_port()
cat("Using port:", port, "\n")

calc_frustum_volume_litres <- function(top_width_cm, base_width_cm, height_cm) {
  a <- top_width_cm
  b <- base_width_cm
  h <- height_cm
  volume_cm3 <- (h / 3) * (a^2 + a*b + b^2)
  volume_litres <- volume_cm3 / 1000
  return(volume_litres)
}

### equivalent of cmd line
no_plant_pot <- FALSE   
leaf_area <- 25.0       
soil_resp_correction <- 0.0
####

if (no_plant_pot) {
  lunchbox_volume <- 1.0
  area_basis <- FALSE
  la <- 1.0
} else {
  pot_volume <- calc_frustum_volume_litres(5.0, 3.4, 5.3)
  lunchbox_volume <- 1.0 - pot_volume
  area_basis <- TRUE
  la <- ifelse(leaf_area > 0, leaf_area, 25.0)
}


logger <- LunchboxLogger(port = port, baud = 9600, 
                         lunchbox_volume = lunchbox_volume, 
                         temp_c = 25.0, leaf_area_cm2 = la,
                         window_size = 41L,       
                         measure_interval = 1L, timeout = 1.0,
                         smoothing = TRUE, 
                         rolling_regression = FALSE,  
                         area_basis = TRUE, 
                         soil_resp_correction = soil_resp_correction)

max_len <- 10 * 60 / logger$measure_interval  # 10 min window

ui <- fluidPage(
  plotOutput("anet_plot", height = "400px"),
  fluidRow(
    column(3, wellPanel(textOutput("co2_text"))),
    column(4, wellPanel(textOutput("anet_text")))
  )
)

server <- function(input, output, session) {
  
  # Store all time series data in reactiveValues
  vals <- reactiveValues(
    xs = numeric(0),
    ys_anet = numeric(0),
    ys_lower = numeric(0),
    ys_upper = numeric(0),
    co2_latest = NA,
    anet_latest = NA
  )
  
  observe({
    invalidateLater(1000, session)  # every second
    
    res <- tryCatch({
      logger$read_and_update()
    }, error = function(e) NULL)
    
    if (!is.null(res)) {
      # Append new data
      vals$xs <- c(vals$xs, res$elapsed_min)
      vals$ys_anet <- c(vals$ys_anet, res$anet)
      vals$ys_lower <- c(vals$ys_lower, res$anet_lower)
      vals$ys_upper <- c(vals$ys_upper, res$anet_upper)
      
      # Keep at max_len points
      if(length(vals$xs) > max_len) {
        vals$xs <- tail(vals$xs, max_len)
        vals$ys_anet <- tail(vals$ys_anet, max_len)
        vals$ys_lower <- tail(vals$ys_lower, max_len)
        vals$ys_upper <- tail(vals$ys_upper, max_len)
      }
      
      vals$co2_latest <- res$co2
      vals$anet_latest <- res$anet
    }
  })
  
  output$co2_text <- renderText({
    if (is.na(vals$co2_latest)) return("Waiting for CO2 data...")
    paste0("CO₂ = ", round(vals$co2_latest), " ppm")
  })
  
  output$anet_text <- renderText({
    if (is.na(vals$anet_latest)) return("")
    units <- ifelse(logger$area_basis, "μmol m⁻² s⁻¹", "μmol box⁻¹ s⁻¹")
    paste0("A_net = ", sprintf("%+.2f", vals$anet_latest), " ", units)
  })
  
  output$anet_plot <- renderPlot({
    if(length(vals$xs) == 0) return(NULL)
    
    df <- data.frame(
      elapsed_min = vals$xs,
      anet = vals$ys_anet,
      anet_lower = vals$ys_lower,
      anet_upper = vals$ys_upper
    )
    
    # Limit x axis window like python script
    latest_time <- max(df$elapsed_min)
    window_start <- max(0, latest_time - 10)
    df <- df[df$elapsed_min >= window_start, ]
    
    # Auto y-limits with margin
    lower <- min(df$anet_lower, na.rm = TRUE)
    upper <- max(df$anet_upper, na.rm = TRUE)
    yrange <- upper - lower
    if (yrange < 1) {
      mid <- (upper + lower) / 2
      ylim <- c(max(-10, mid - 1), mid + 1)
    } else {
      margin <- yrange * 0.1
      ylim <- c(max(-10, lower - margin), upper + margin)
    }
    
    ggplot(df, aes(x = elapsed_min)) +
      geom_ribbon(aes(ymin = anet_lower, ymax = anet_upper),
                  fill = "#0b5345", alpha = 0.2) +
      geom_line(aes(y = anet, group = 1), color = "#28b463", size = 1.25) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "darkgrey") +
      coord_cartesian(xlim = c(window_start, window_start + 10), ylim = ylim) +
      labs(
        x = "Elapsed Time (min)",
        y = ifelse(logger$area_basis,
                   "Net assimilation rate (μmol m⁻² s⁻¹)",
                   "Net assimilation rate (μmol box⁻¹ s⁻¹)")
      ) +
      theme_minimal(base_size = 15, base_family = "Helvetica") +
      theme(
        plot.title = element_text(face = "bold", size = 18, margin = margin(b = 10)),
        axis.title = element_text(size = 16, margin = margin(t = 10, r = 10, b = 10, l = 10)),
        axis.text = element_text(color = "gray20"),
        panel.grid.major.y = element_line(color = "gray85", size = 0.5),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        plot.background = element_rect(fill = "white", color = NA),
        panel.border = element_rect(color = "gray80", fill = NA, size = 0.5)
      )
  })
  
  
  session$onSessionEnded(function() {
    logger$close()
  })
}

shinyApp(ui, server)
