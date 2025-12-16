import React, { useState, useEffect, useCallback, useMemo, memo } from "react";
import { Button, Image, ScrollView, Text, View, XStack, YStack } from "tamagui";
import {
  User,
  Settings,
  LogOut,
  Camera,
  Edit3,
  Share,
} from "@tamagui/lucide-icons";
import { StatusBar, FlatList, RefreshControl } from "react-native";
import { ResponsiveContainer } from "../../components/ResponsiveContainer";
import { ShinyText } from "../../components/ShinyText";
import { FleekButton } from "../../components/ui";
import { FullscreenImageViewer } from "../../components/FullscreenImageViewer";
import { LoadingImage, GridShimmer } from "../../components/image";
import { authService } from "../../services/auth";
import { useQuery } from "@tanstack/react-query";
import { GeneratedImage } from "../../types/shared";
import { API_BASE_URL } from "../../hooks/useContentGeneration";
import { useFeed, FeedPost } from "../../hooks/useFeed";
import { useAuthFlow } from "../../hooks/useAuthFlow";
import { LinearGradient } from "expo-linear-gradient";

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
            minHeight={120}
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
            if (item.status === "ready") {
              onPress(item);
            }
          }}
          disabled={item.status !== "ready"}
        >
          <LoadingImage
            source={{ uri: item.url || "" }}
            width="100%"
            aspectRatio={1}
            resizeMode="cover"
            status={item.status || "ready"}
          />
        </Button>
      </YStack>
    );
  }
);

export default function ProfilePage() {
  const { user } = useAuthFlow();
  const [isLoading, setIsLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState<GeneratedImage | null>(
    null
  );
  const [showFullscreen, setShowFullscreen] = useState(false);

  // Image handling functions
  const handleImagePress = useCallback((image: GeneratedImage) => {
    setSelectedImage(image);
    setShowFullscreen(true);
  }, []);

  const handleCloseFullscreen = useCallback(() => {
    setShowFullscreen(false);
    setSelectedImage(null);
  }, []);

  const {
    posts: feedPosts,
    isLoading: isLoadingImages,
    refreshFeed,
  } = useFeed();

  const userSharedImages = useMemo(() => {
    return feedPosts.filter((post: FeedPost) => {
      return post.user_id === user?.id;
    });
  }, [feedPosts]);

  const images = useMemo(() => {
    return userSharedImages.flatMap((post: FeedPost) => {
      return post.images.map((img: any) => ({
        url: img.url,
        prompt: post.prompt || "No prompt",
        model: "flux-dev",
        dimensions: `${img.width || 1024}x${img.height || 1024}`,
        createdAt: post.created || new Date().toISOString(),
        contentId: post.content_id,
        status: "ready" as const,
      }));
    });
  }, [userSharedImages]);

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

  const refreshHandler = useCallback(async () => {
    await refreshFeed();
  }, [refreshFeed]);

  return (
    <ResponsiveContainer backgroundColor="#111111">
      <StatusBar barStyle="light-content" backgroundColor="#111111" />

      <ScrollView flex={1} showsVerticalScrollIndicator={false}>
        <LinearGradient
          colors={["rgba(66, 66, 66, 0.78)", "#111111"]}
          style={{
            width: "100%",
          }}
        >
          <YStack paddingTop="$12" paddingBottom="$6" paddingHorizontal="$4">
            {/* Avatar and User Info */}
            <YStack alignItems="flex-start" gap="$4">
              {/* Avatar */}
              <LinearGradient
                colors={["rgba(180, 180, 180, 0.78)", "rgba(37, 33, 29, 0.78)"]}
                style={{
                  width: 60,
                  height: 60,
                  borderRadius: 999,
                  alignItems: "center",
                  justifyContent: "center",
                  borderWidth: 1,
                  borderColor: "rgba(198, 198, 198, 0.1)",
                }}
              >
                <User size={24} color="#ffffff" />
              </LinearGradient>

              {/* User Info */}
              <YStack alignItems="flex-start" gap="$2">
                <Text color="white" fontSize="$4" fontWeight="700">
                  Anonymous
                </Text>
                <Text color="white" fontSize="$3" opacity={0.8}>
                  {images.length} images shared
                </Text>
              </YStack>
            </YStack>
          </YStack>
        </LinearGradient>

        {/* Shared Images Section */}
        <YStack flex={1} backgroundColor="#111111" paddingTop="$4">
          {/* Images Grid */}
          {isLoadingImages ? (
            <YStack paddingHorizontal="$4" gap="$2">
              <GridShimmer count={9} />
            </YStack>
          ) : images.length === 0 ? (
            <YStack
              flex={1}
              alignItems="center"
              justifyContent="center"
              paddingVertical="$12"
              gap="$3"
            >
              <View
                width={80}
                height={80}
                borderRadius="$12"
                backgroundColor="#333"
                alignItems="center"
                justifyContent="center"
              >
                <Share size={32} color="#666" />
              </View>
              <Text
                color="#666"
                fontSize="$4"
                textAlign="center"
                fontWeight="600"
              >
                No images shared yet
              </Text>
              <Text color="#444" fontSize="$3" textAlign="center">
                Create and share your first image to see it here
              </Text>
            </YStack>
          ) : (
            <FlatList
              data={flatListData}
              renderItem={renderRow}
              refreshControl={
                <RefreshControl
                  refreshing={isLoadingImages}
                  onRefresh={refreshHandler}
                />
              }
              keyExtractor={(_, index) => index.toString()}
              showsVerticalScrollIndicator={false}
              removeClippedSubviews={true}
              maxToRenderPerBatch={10}
              updateCellsBatchingPeriod={50}
              initialNumToRender={10}
              windowSize={10}
              contentContainerStyle={{ paddingBottom: 100 }}
            />
          )}
        </YStack>
      </ScrollView>

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
