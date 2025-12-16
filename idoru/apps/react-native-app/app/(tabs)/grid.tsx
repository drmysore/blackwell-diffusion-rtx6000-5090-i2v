import React, { useState, useCallback, useEffect, memo, useMemo } from "react";
import { Button, Image, ScrollView, Text, View, XStack, YStack } from "tamagui";
import {
  StatusBar,
  ActivityIndicator,
  FlatList,
  RefreshControl,
} from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { ShinyText } from "../../components/ShinyText";
import { ResponsiveContainer } from "../../components/ResponsiveContainer";
import { FullscreenImageViewer } from "../../components/FullscreenImageViewer";
import { LoadingImage, GridShimmer } from "../../components/image";
import { useAuthFlow } from "../../hooks/useAuthFlow";
import { useContentGeneration } from "../../hooks/useContentGeneration";
import { useImageHandling } from "../../hooks/image";
import { API_BASE_URL } from "../../hooks/useContentGeneration";
import { authService } from "../../services/auth";
import { useQuery } from "@tanstack/react-query";
import { GeneratedImage } from "../../types/shared";

// Components

// Memoized grid item component for better performance
const GridItem = memo(
  ({
    item,
    index,
    onPress,
  }: {
    item: GeneratedImage | null;
    index: number;
    onPress: (img: GeneratedImage) => void;
  }) => {
    if (!item) {
      return (
        <YStack key={index} flex={1} overflow="hidden">
          <View
            flex={1}
            backgroundColor="#1a1a1a"
            borderRadius="$4"
            alignItems="center"
            justifyContent="center"
            minHeight={200}
          >
            <Text color="#444" fontSize="$2" textAlign="center">
              Empty
            </Text>
          </View>
        </YStack>
      );
    }

    return (
      <YStack key={index} flex={1} overflow="hidden">
        <Button
          unstyled
          backgroundColor="transparent"
          overflow="hidden"
          onPress={() => {
            // Only allow fullscreen for ready images
            if (item.status === "ready") {
              onPress(item);
            }
          }}
          disabled={item.status !== "ready"}
        >
          <LoadingImage
            source={{ uri: item.url || "" }}
            width="100%"
            aspectRatio={
              item.dimensions.includes("1024x1024")
                ? 1
                : item.dimensions.includes("768x1024")
                  ? 3 / 4
                  : 4 / 3
            }
            resizeMode="cover"
            status={item.status || "ready"}
          />
        </Button>
      </YStack>
    );
  }
);

export default function GridPage() {
  // Authentication state
  const { state: authState } = useAuthFlow();

  // Content generation hook
  useContentGeneration();

  // Image handling hook
  const {
    selectedImage,
    showFullscreen,
    handleImagePress,
    handleCloseFullscreen,
  } = useImageHandling();

  // TanStack Query for loading content
  const {
    data: contentData,
    isLoading: isLoadingHistory,
    error,
    refetch,
  } = useQuery({
    queryKey: ["userContent"],
    queryFn: async () => {
      try {
        const accessToken = await authService.ensureAuthenticated();
        let currentUserId = null;

        try {
          const tokenParts = accessToken.split(".");
          if (tokenParts.length === 3) {
            const base64Payload = tokenParts[1];
            const decodedPayload =
              typeof atob !== "undefined"
                ? atob(base64Payload)
                : Buffer.from(base64Payload, "base64").toString("utf-8");
            const payload = JSON.parse(decodedPayload);
            currentUserId = payload.sub;
          }
        } catch (tokenError) {
          console.warn("⚠️ Could not extract user ID from token:", tokenError);
        }

        const headers = {
          "Content-Type": "application/json",
          Authorization: `Bearer ${accessToken}`,
        };

        const response = await fetch(
          `${API_BASE_URL}/content/?limit=50&status=ready&user_id=${currentUserId}`,
          {
            headers,
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        console.log(
          "📜 User-specific content loaded:",
          data.items?.length || 0,
          "items"
        );

        let feedPosts: any[] = [];

        if (Array.isArray(data)) {
          feedPosts = data;
        } else if (data.items && Array.isArray(data.items)) {
          feedPosts = data.items;
        } else if (data.content && Array.isArray(data.content)) {
          feedPosts = data.content;
        } else if (data.data && Array.isArray(data.data)) {
          feedPosts = data.data;
        }

        const historyImages: GeneratedImage[] = feedPosts
          .reverse()
          .flatMap((post: any) => {
            if (!post.images && post.status === "generating") {
              return [
                {
                  url: "",
                  prompt: post.prompt || "No prompt",
                  model: post.generation_model || "flux-dev",
                  dimensions: "1024x1024",
                  createdAt: post.created || new Date().toISOString(),
                  contentId: post.id || post.content_id,
                  status: post.status || "generating",
                  requestId: post.job_id,
                },
              ];
            }

            return (
              post.images?.map((img: any) => ({
                url: img.url,
                prompt: post.prompt || "No prompt",
                model: post.generation_model || "flux-dev",
                dimensions: `${img.width || 1024}x${img.height || 1024}`,
                createdAt: post.created || new Date().toISOString(),
                contentId: post.id || post.content_id,
                status: post.status || "ready",
              })) || []
            );
          });

        return historyImages;
      } catch (error) {
        console.error("❌ Failed to load content:", error);
        // Fallback to local storage
        const historyKey = await getChatHistoryKey();
        const savedHistory = await AsyncStorage.getItem(historyKey);
        if (savedHistory) {
          return JSON.parse(savedHistory);
        }
        return [];
      }
    },
    staleTime: 30 * 1000, // 30 seconds
    cacheTime: 5 * 60 * 1000, // 5 minutes
    retry: 2,
  });

  const images = contentData || [];

  // Prepare data for FlatList (group images in sets of 3 for 3-column layout)
  const flatListData = useMemo(() => {
    const rows: (GeneratedImage | null)[][] = [];
    for (let i = 0; i < images.length; i += 3) {
      rows.push([
        images[i] || null,
        images[i + 1] || null,
        images[i + 2] || null,
      ]);
    }
    return rows;
  }, [images]);

  // Render function for FlatList
  const renderRow = useCallback(
    ({ item: row }: { item: (GeneratedImage | null)[] }) => (
      <XStack justifyContent="space-between">
        {row.map((img, colIndex) => (
          <GridItem
            key={colIndex}
            item={img}
            index={colIndex}
            onPress={handleImagePress}
          />
        ))}
      </XStack>
    ),
    [handleImagePress]
  );

  // Chat history persistence helper
  const getChatHistoryKey = useCallback(async () => {
    try {
      const token = await authService.getAccessToken();
      // Simple user ID extraction from token or use timestamp-based ID
      const userId = token ? `user_${token.slice(-8)}` : "anonymous";
      return `chat_history_${userId}`;
    } catch {
      return "chat_history_anonymous";
    }
  }, []);

  const refreshGridHandler = useCallback(async () => {
    await refetch();
  }, [refetch]);

  return (
    <ResponsiveContainer backgroundColor="#111111">
      <StatusBar barStyle="light-content" backgroundColor="#111111" />

      {/* Grid View */}
      {isLoadingHistory ? (
        <YStack flex={1} paddingTop="$4">
          <YStack gap="$2">
            <GridShimmer count={12} />
          </YStack>
        </YStack>
      ) : images.length === 0 ? (
        <YStack
          flex={1}
          alignItems="center"
          justifyContent="center"
          paddingVertical="$8"
          gap="$3"
        >
          <Text color="#666" fontSize="$4" textAlign="center">
            No images yet
          </Text>
          <Text color="#444" fontSize="$3" textAlign="center">
            Create your first image to see it here
          </Text>
        </YStack>
      ) : (
        <YStack flex={1}>
          <FlatList
            data={flatListData}
            renderItem={renderRow}
            refreshControl={
              <RefreshControl
                refreshing={isLoadingHistory}
                onRefresh={refreshGridHandler}
              />
            }
            keyExtractor={(_, index) => index.toString()}
            showsVerticalScrollIndicator={false}
            removeClippedSubviews={true}
            maxToRenderPerBatch={10}
            updateCellsBatchingPeriod={50}
            initialNumToRender={10}
            windowSize={10}
          />
        </YStack>
      )}

      {/* Fullscreen Image Viewer */}
      {selectedImage && (
        <FullscreenImageViewer
          imageUrl={selectedImage.url}
          isVisible={showFullscreen}
          contentId={selectedImage.contentId}
          onClose={handleCloseFullscreen}
          post={{
            prompt: selectedImage.prompt,
            model: selectedImage.model,
            dimensions: selectedImage.dimensions,
          }}
        />
      )}
    </ResponsiveContainer>
  );
}
