library(shiny)
library(plotly)
library(readr)
library(dplyr)
library(lubridate)

calc_volume_litres <- function(width_cm, height_cm, length_cm) {
  volume_cm3 <- width_cm * height_cm * length_cm
  return(volume_cm3 / 1000)
}

calc_anet <- function(delta_ppm_s, lunchbox_volume_litres, temp_k = 295.15) {
  pressure <- 101325.0  # Pa
  rgas <- 8.314         # J/(K mol)
  volume_m3 <- lunchbox_volume_litres / 1000.0
  an_leaf <- (delta_ppm_s * pressure * volume_m3) / (rgas * temp_k)
  return(an_leaf)
}

find_latest_csv <- function(desktop_path = "~/Desktop", prefix = "PAS_CO2_datalog_") {
  files <- list.files(path = path.expand(desktop_path), pattern = paste0("^", prefix, ".*\\.csv$"),
                      full.names = TRUE)
  if (length(files) == 0) return(NULL)
  files[which.max(file.info(files)$mtime)]
}

load_and_process_data <- function(fname, lunchbox_volume_litres) {
  if (!file.exists(fname)) return(NULL)
  
  df <- read_csv(fname, comment = "#", show_col_types = FALSE)
  
  names(df)[3] <- "co2"
  names(df)[1:2] <- c("timestamp", "time")
  
  df <- df %>%
    mutate(
      time = ymd_hms(time),
      timestamp = as.numeric(timestamp),
      delta_seconds = c(NA, diff(timestamp)),
      delta_ppm = c(NA, diff(co2)),
      delta_ppm_s = delta_ppm / delta_seconds,
      anet = sapply(delta_ppm_s, function(x) if (!is.na(x)) calc_anet(x, lunchbox_volume_litres) else NA)
    ) %>%
    arrange(time)
  
  return(df)
}


ui <- fluidPage(
  plotlyOutput("plot"),
  tags$script(HTML("setInterval(function() { Shiny.setInputValue('auto_refresh', new Date()); }, 10000);"))
)


server <- function(input, output, session) {
  lunchbox_width_cm <- 17.5
  lunchbox_height_cm <- 5
  lunchbox_length_cm <- 12
  temp_k <- 295.15
  lunchbox_volume <- calc_volume_litres(lunchbox_width_cm, lunchbox_height_cm, lunchbox_length_cm)
  
  get_data <- reactive({
    fname <- find_latest_csv()
    if (is.null(fname)) return(NULL)
    load_and_process_data(fname, lunchbox_volume)
  })
  
  output$plot <- renderPlotly({
    req(input$auto_refresh)
    df <- get_data()
    req(!is.null(df), nrow(df) > 1)
    
    plot_ly(df, x = ~time) %>%
      add_lines(y = ~co2, name = "CO₂ (ppm)", line = list(color = "green"), yaxis = "y1") %>%
      add_lines(y = ~anet, name = "Anet (μmol s⁻¹)", yaxis = "y2") %>%
      layout(
        xaxis = list(title = "Time"),
        yaxis = list(title = "CO₂ (ppm)", side = "left"),
        yaxis2 = list(title = "Net Assimilation Rate (μmol s⁻¹)", overlaying = "y", side = "right"),
        legend = list(
          x = 1, y = 0,
          xanchor = "right", yanchor = "bottom",
          bgcolor = 'rgba(255,255,255,0.7)',
          bordercolor = 'black',
          borderwidth = 1
        ),
        margin = list(r = 150)
      )
  })
}

shinyApp(ui, server)
