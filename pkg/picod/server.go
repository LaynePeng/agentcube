package picod

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

var startTime = time.Now() // Server start time

// Config defines server configuration
type Config struct {
	Port int `json:"port"`
}

// Server defines the PicoD HTTP server
type Server struct {
	engine *gin.Engine
	config Config
}

// NewServer creates a new PicoD server instance
func NewServer(config Config) *Server {
	// Disable Gin debug output in production mode
	gin.SetMode(gin.ReleaseMode)

	engine := gin.New()

	// Global middleware
	engine.Use(gin.Logger())   // Request logging
	engine.Use(gin.Recovery()) // Crash recovery

	// API route group
	api := engine.Group("/api")
	{
		api.POST("/execute", ExecuteHandler)
		api.POST("/files", UploadFileHandler)
		api.GET("/files/*path", DownloadFileHandler)
	}

	// Health check (no authentication required)
	engine.GET("/health", HealthCheckHandler)

	return &Server{
		engine: engine,
		config: config,
	}
}

// Run starts the server
func (s *Server) Run() error {
	addr := fmt.Sprintf(":%d", s.config.Port)
	log.Printf("PicoD server starting on %s", addr)
	return http.ListenAndServe(addr, s.engine)
}

// HealthCheckHandler handles health check requests
func HealthCheckHandler(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"service": "PicoD",
		"version": "1.0.0",
		"uptime":  time.Since(startTime).String(),
	})
}
